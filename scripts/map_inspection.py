"""Map an inspection pass: where on the net was each detection, and what was missed.

Turns a pass from a list of frames into a **map**. Every detection gets a
position in metres along and across the swept strip, with a physical size and a
drift estimate, and the pass gets a coverage report naming the bands of net that
went unphotographed.

Pipeline
--------
1. Join frames to ROV telemetry on the bag clock (:mod:`netinspect.frame_sync`).
2. Estimate frame-to-frame motion by feature matching.
3. Calibrate millimetres-per-pixel from telemetry travel — no chessboard.
4. Integrate motion into net-frame positions with a drift estimate.
5. Run a detector and place each detection on the net.
6. Report coverage, gaps, and a cross-check of the visual path length against
   telemetry.

Honesty
-------
This maps the **inspected strip**, not a pen. One clip's arc is far too straight
to identify a pen radius, so nothing here reconstructs a whole net. Positions
drift with distance from the start and are reported with an error bar rather
than as points. And the detections being mapped are, on SOLAQUA, all false
positives — the net is undamaged. The map is the mechanism; the damage is
synthetic.

Examples
--------
    python scripts/map_inspection.py --clip 2024-08-22_14-29-05
    python scripts/map_inspection.py --clip 2024-08-22_14-29-05 --method seg_gpu --no-figure
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect import frame_sync as F
from netinspect import mapping as M
from netinspect import telemetry as T
from netinspect.inference import NetInspector
from netinspect.utils import ensure_dir, get_logger, image_size, read_image, write_json

LOGGER = get_logger()

FRAME_DIRS = {
    "2024-08-22_14-06-43": "data/processed/solaqua_frames_dense",
    "2024-08-22_14-47-39": "data/processed/solaqua_bag2",
    "2024-08-22_14-29-05": "data/processed/solaqua_bag3",
    "2024-08-20_15-18-27": "data/processed/solaqua_diffday",
}
WEIGHTS = {"det_v1": "models/yolo_damage_v1.pt",
           "seg_v3": "models/yolo_damage_seg_v3.pt",
           "seg_gpu": "models/yolo_damage_seg_gpu.pt"}


def make_figure(track, dets, sites, cov, clip, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK_SOFT, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8ae"
    PATH, ALARM = "#2a78d6", "#eb6834"

    along = np.array([p.along_m for p in track])
    across = np.array([p.across_m for p in track])
    fp = np.array([p.mm_per_px * 1280 / 1000.0 for p in track])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.2), facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(top=0.80, hspace=0.62, left=0.075, right=0.97)

    # Swept strip + detections.
    ax.set_facecolor(SURFACE)
    ax.fill_between(along, across - fp / 2, across + fp / 2, color=PATH, alpha=0.13,
                    label="camera footprint (swept)")
    ax.plot(along, across, color=PATH, lw=1.8, label="ROV path")
    for g in cov.gaps:
        ax.axvspan(g["from_along_m"], g["to_along_m"], color=ALARM, alpha=0.18)
    if dets:
        ax.scatter([d.along_m for d in dets], [d.across_m for d in dets],
                   s=10, c=NEUTRAL, alpha=0.55, zorder=4,
                   label=f"per-frame detections ({len(dets)})")
    if sites:
        ax.scatter([s_.along_m for s_ in sites], [s_.across_m for s_ in sites],
                   s=[60 + 6 * s_.sightings for s_ in sites],
                   facecolor="none", edgecolor=ALARM, linewidth=2.4, zorder=6,
                   label=f"distinct sites ({len(sites)})")
        for s_ in sites:
            if s_.sightings >= 3:
                ax.annotate(f"{s_.sightings}x", (s_.along_m, s_.across_m),
                            textcoords="offset points", xytext=(0, 14),
                            ha="center", fontsize=8.5, color=ALARM, fontweight="bold")
    ax.set_xlabel("metres along the sweep", fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel("metres across", fontsize=9.5, color=INK_SOFT)
    ax.grid(alpha=0.3, color=NEUTRAL)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.38), labelcolor=INK_SOFT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("A · The inspected strip, in the net's own frame",
                 fontsize=11, color=INK, fontweight="bold", loc="left", pad=8)

    # Standoff and drift along the pass.
    ax2.set_facecolor(SURFACE)
    so = [p.standoff_m for p in track]
    ax2.plot(along, so, color=PATH, lw=1.6, label="standoff (m)")
    ax2.fill_between(along, [p.drift_m for p in track], 0,
                     color=ALARM, alpha=0.25, label="position uncertainty (m)")
    ax2.set_xlabel("metres along the sweep", fontsize=9.5, color=INK_SOFT)
    ax2.set_ylabel("metres", fontsize=9.5, color=INK_SOFT)
    ax2.grid(alpha=0.3, color=NEUTRAL)
    # Standoff climbs towards the end of the pass, so the only reliably empty
    # corner is upper-left; headroom keeps the legend off the line.
    ax2.set_ylim(top=max(so) * 1.38)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2,
               labelcolor=INK_SOFT)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("B · Standoff, and how position confidence decays with distance",
                  fontsize=11, color=INK, fontweight="bold", loc="left", pad=8)

    fig.suptitle(f"Inspection map — {clip}", fontsize=13.5, color=INK,
                 fontweight="bold", x=0.075, ha="left", y=0.96)
    fig.text(0.075, 0.90,
             f"{cov.along_extent_m:.1f} m of net swept · {cov.swept_area_m2:.1f} m² footprint · "
             f"{len(cov.gaps)} coverage gap(s) shown in orange · "
             f"positions drift ~{M.DRIFT_FRACTION:.0%} of distance travelled.\n"
             "Scale is self-calibrated from telemetry travel, not a calibration target. "
             "This is the inspected STRIP — not a model of a pen. On SOLAQUA the net is "
             "undamaged, so every detection plotted here is a false positive.",
             fontsize=8.8, color=INK_SOFT, ha="left", va="top", linespacing=1.5)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figure_3d(track, sites, cov, clip, out_path):
    """Where the camera actually flew, relative to the net wall.

    The three axes are all measured, none reconstructed: along-track from visual
    odometry, standoff from the net-plane sensor, depth from the pressure sensor.
    The wall is drawn flat because that is what the data supports — see the
    caption. This answers "where was the vehicle looking from", which is the
    question that decides whether a clean result means anything.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK_SOFT, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8ae"
    PATH, ALARM, NET = "#2a78d6", "#eb6834", "#9aa79c"

    along = np.array([p.along_m for p in track])
    stand = np.array([p.standoff_m for p in track], dtype=float)
    depth = np.array([p.depth_m if p.depth_m is not None else np.nan for p in track])
    if not np.isfinite(depth).any():
        depth = np.zeros_like(along)

    fig = plt.figure(figsize=(13, 5.4), facecolor=SURFACE)
    ax = fig.add_subplot(111, projection="3d", facecolor=SURFACE)
    # A 3-D axes reserves a square region regardless of the box aspect, so the
    # rect is overdrawn deliberately to squeeze out the dead space around a wide,
    # flat scene.
    ax.set_position([-0.11, -0.21, 1.16, 1.27])

    d0, d1 = float(np.nanmin(depth)) - 0.2, float(np.nanmax(depth)) + 0.2
    a0, a1 = float(along.min()), float(along.max())
    s_max = max(1.0, float(np.nanmax(stand)) * 1.1)

    # The pass is 5.5 m long, ~1 m deep in standoff and 0.6 m in depth. A cubic
    # box would stretch the thin axes and make the flight look far more erratic
    # than it was, so the box carries the real proportions.
    ax.set_box_aspect(((a1 - a0), s_max * 1.6, (d1 - d0) * 2.2))

    # The net wall at standoff 0: a faint surface so it reads as a solid boundary
    # the vehicle is flying alongside, with mesh lines on top so it reads as
    # netting rather than a pane of glass.
    wa, wd = np.meshgrid(np.linspace(a0, a1, 2), np.linspace(d0, d1, 2))
    ax.plot_surface(wa, np.zeros_like(wa), wd, color=NET, alpha=0.16,
                    shade=False, zorder=0, linewidth=0)
    for a in np.linspace(a0, a1, 34):
        ax.plot([a, a], [0, 0], [d0, d1], color=NET, lw=0.7, alpha=0.8, zorder=1)
    for d in np.linspace(d0, d1, 8):
        ax.plot([a0, a1], [0, 0], [d, d], color=NET, lw=0.7, alpha=0.8, zorder=1)

    # The flight path, and droplines showing the standoff held at each sample.
    ax.plot(along, stand, depth, color=PATH, lw=2.0, zorder=6, label="ROV path")
    for i in range(0, len(along), max(1, len(along) // 45)):
        ax.plot([along[i], along[i]], [stand[i], 0], [depth[i], depth[i]],
                color=PATH, lw=0.6, alpha=0.3, zorder=4)

    # Sites live ON the wall, not on the path.
    for s_ in sites:
        z = s_.depth_m if s_.depth_m is not None else float(np.nanmean(depth))
        ax.scatter([s_.along_m], [0], [z], s=40 + 7 * s_.sightings,
                   facecolor="none", edgecolor=ALARM, linewidth=2.2, zorder=8)
        if s_.sightings >= 3:
            ax.text(s_.along_m, 0, z - 0.09, f"{s_.sightings}x", color=ALARM,
                    fontsize=8.5, fontweight="bold", ha="center", zorder=9)
    if sites:
        ax.scatter([], [], [], facecolor="none", edgecolor=ALARM, linewidth=2.2,
                   s=70, label=f"defect sites on the wall ({len(sites)})")
    ax.plot([], [], color=NET, lw=1.0, label="net wall (standoff = 0)")

    ax.set_xlabel("metres along the sweep", fontsize=9.5, color=INK_SOFT, labelpad=14)
    ax.set_ylabel("standoff from net (m)", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_zlabel("depth (m)", fontsize=9.5, color=INK_SOFT, labelpad=2)
    ax.set_xlim(a0, a1)
    ax.set_ylim(s_max, 0)                    # net wall sits at the BACK, behind the path
    ax.set_zlim(d1, d0)                      # deeper is down
    ax.view_init(elev=20, azim=-58)
    ax.grid(alpha=0.25)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor(NEUTRAL)
        pane.pane.set_alpha(0.35)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5)
    ax.zaxis.set_tick_params(pad=1)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=9,
               loc="upper right", bbox_to_anchor=(0.985, 0.985), labelcolor=INK_SOFT)

    fig.suptitle(f"Where the camera flew — {clip}", fontsize=13.5, color=INK,
                 fontweight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.925,
             f"All three axes are measured, none reconstructed: along-track from visual "
             f"odometry, standoff from the net-plane sensor, depth from the pressure "
             f"sensor.\nThe pass held {np.nanmin(stand):.2f}–{np.nanmax(stand):.2f} m off "
             f"the net across {cov.along_extent_m:.1f} m at a near-constant "
             f"{np.nanmean(depth):.1f} m depth — one horizontal band, not a whole pen.\n"
             "The wall is drawn FLAT on purpose. Over this arc the deviation from a "
             "straight line is inside USBL noise, so curving it would be inventing "
             "geometry rather than measuring it.",
             fontsize=8.8, color=INK_SOFT, ha="left", va="top", linespacing=1.5)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="2024-08-22_14-29-05", choices=list(FRAME_DIRS))
    ap.add_argument("--method", default="det_v1", choices=list(WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--merge-radius", type=float, default=0.25,
                    help="metres within which sightings are the same physical site")
    ap.add_argument("--out", default="reports/results/inspection_maps")
    ap.add_argument("--figure", default="docs/images/inspection_map.png")
    ap.add_argument("--figure-3d", default="docs/images/inspection_map_3d.png")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    idx = F.load_frame_index(args.clip)
    if idx is None:
        raise SystemExit(f"No frame index for {args.clip} — build it first "
                         "(see scripts/analyze_operating_envelope.py).")
    tele = T.extract_telemetry(f"data/raw/solaqua/{args.clip}_data.bag",
                               streams=["net_plane", "depth_temp"])
    joined = F.join_frames(FRAME_DIRS[args.clip], tele, {args.clip: idx}, tolerance_s=0.5)
    joined = joined[joined["net_plane_net_distance"].notna()].reset_index(drop=True)
    if args.max_frames:
        joined = joined.head(args.max_frames)
    if len(joined) < 10:
        raise SystemExit("Too few frames with telemetry to map.")

    paths = joined["path"].tolist()
    LOGGER.info("Estimating motion across %d frames…", len(paths))
    motions = M.track_motion(paths)
    matched = [m for m in motions if m is not None]
    LOGGER.info("Matched %d/%d pairs (mean %.0f%% inliers)", len(matched), len(motions),
                100 * np.mean([m.inlier_ratio for m in matched]) if matched else 0)

    dt = np.diff(joined["t"].to_numpy())
    speed = np.hypot(joined["net_plane_net_vel_u"], joined["net_plane_net_vel_v"]).to_numpy()
    tele_step = speed[1:] * dt
    px_step = np.array([m.magnitude_px if m is not None else np.nan for m in motions])
    standoff = joined["net_plane_net_distance"].to_numpy()

    w, h = image_size(paths[0])
    scale = M.calibrate_scale(px_step, tele_step, standoff[1:], image_width_px=w)
    LOGGER.info("Scale: %.3f mm/px at 1 m (implied HFOV %s°)",
                scale.mm_per_px_at_1m, scale.implied_hfov_deg)

    depth = joined["depth_temp_depth"].to_numpy() if "depth_temp_depth" in joined else None
    track = M.build_track(paths, joined["frame"].tolist(), joined["t"].tolist(),
                          motions, standoff_m=standoff, depth_m=depth, scale=scale)

    LOGGER.info("Running %s over the pass…", args.method)
    insp = NetInspector(yolo_weights=WEIGHTS[args.method])
    by_frame: dict[str, list] = {}
    for name, p in zip(joined["frame"], paths):
        r = insp.predict(read_image(p), method="yolo", conf=args.conf)
        if r.boxes:
            by_frame[name] = [{"bbox": [b.x1, b.y1, b.x2, b.y2], "score": b.score}
                              for b in r.boxes]
    dets = M.localise_detections(track, by_frame, (w, h))
    sites = M.cluster_sites(dets, radius_m=args.merge_radius)
    cov = M.coverage(track, (w, h))
    check = M.validate_against_telemetry(track, float(np.nansum(tele_step)))

    payload = {
        "clip": args.clip, "method": args.method, "conf": args.conf,
        "frames": len(track),
        "motion": {"pairs": len(motions), "matched": len(matched),
                   "mean_inlier_ratio": round(float(np.mean(
                       [m.inlier_ratio for m in matched])), 3) if matched else None},
        "scale": scale.to_dict(),
        "coverage": cov.to_dict(),
        "telemetry_check": check,
        "sites": [s.to_dict() for s in sites],
        "detections": [d.to_dict() for d in dets],
        "track": [p.to_dict() for p in track],
        "caveats": [
            "Maps the inspected strip, not a pen. One clip's arc is too straight "
            "to identify a pen radius.",
            "Positions are relative to the start of the pass and drift with "
            "distance; each carries an uncertainty estimate.",
            "Scale is self-calibrated from telemetry travel. Check the implied "
            "field of view before trusting it.",
            "SOLAQUA nets are undamaged, so every detection mapped here is a "
            "false positive. Recall on real damage is unmeasured.",
            "Sighting count is evidence that a location is a distinct object, "
            "NOT evidence that it is damage. A well-observed false positive is "
            "still a false positive.",
        ],
    }
    out_dir = ensure_dir(args.out)
    write_json(payload, out_dir / f"{args.clip}_map.json")

    print("\n" + "=" * 78)
    print(f"INSPECTION MAP — {args.clip}")
    print("=" * 78)
    print(f"  frames              {len(track)} ({len(matched)}/{len(motions)} pairs matched)")
    if matched:
        print(f"  match quality       {100*np.mean([m.inlier_ratio for m in matched]):.0f}% inliers")
    print(f"  scale               {scale.mm_per_px_at_1m:.3f} mm/px at 1 m "
          f"→ {scale.mm_per_px(scale.reference_standoff_m):.2f} mm/px at "
          f"{scale.reference_standoff_m:.2f} m")
    print(f"  implied HFOV        {scale.implied_hfov_deg}°   <- sanity check")
    print(f"  swept               {cov.along_extent_m:.2f} m along · "
          f"{cov.mean_footprint_m:.2f} m footprint · {cov.swept_area_m2:.2f} m²")
    print(f"  coverage gaps       {len(cov.gaps)}")
    for g in cov.gaps[:5]:
        print(f"      {g['gap_m']:.2f} m between {g['from_along_m']:.2f} and "
              f"{g['to_along_m']:.2f} m along")
    print(f"  telemetry check     visual {check.get('visual_path_m')} m vs "
          f"telemetry {check.get('telemetry_path_m')} m "
          f"(ratio {check.get('ratio')})")
    print(f"\n  per-frame detections  {len(dets)}")
    print(f"  DISTINCT SITES        {len(sites)}  "
          f"(sightings merged within {args.merge_radius} m)")
    for site in sites[:8]:
        print(f"      {site.describe()}")
    if dets and sites:
        strong = [s for s in sites if s.sightings >= 3]
        print(f"\n  -> {len(dets)} alerts became {len(sites)} things to look at "
              f"({len(dets) / len(sites):.0f}x fewer); {len(strong)} seen in 3+ frames.")
        print("     Sighting count is evidence of a DISTINCT OBJECT, not of damage:")
        print("     a well-observed false positive is still a false positive.")

    if not args.no_figure and track:
        fig = make_figure(track, dets, sites, cov, args.clip, Path(args.figure))
        print(f"\nWrote {fig}")
        fig3 = make_figure_3d(track, sites, cov, args.clip, Path(args.figure_3d))
        print(f"Wrote {fig3}")
    print(f"Wrote {out_dir}/{args.clip}_map.json")
    print("\nCaveat: the inspected strip, not a pen. On SOLAQUA every mapped "
          "detection is a\nfalse positive — the net is undamaged.")


if __name__ == "__main__":
    main()

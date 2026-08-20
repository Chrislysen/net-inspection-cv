"""WHY does the detector fire on bag3? Measure the false alarms instead of guessing.

Three strategies have now failed to close the between-clip false-alarm spread —
photometric augmentation, the Jerlov water model, and hard-negative mining on the
offending clip itself (`eval_hard_negatives.py`), the last of which made it
measurably worse. Feeding the model more pixels is not working, so the next
honest step is to characterise what it is actually firing on.

The repo has asserted for a while that the culprit is "thin bright mooring cords
rigged around the calibration markers". That came from looking at frames, which
is a reasonable way to form a hypothesis and a poor way to support one. This
script tests it.

The hypothesis, stated so it can fail
-------------------------------------
The detector was trained on **synthetic** damage: `netinspect.synthetic` paints
holes as compact dark ellipses and tears as elongated dark streaks
(`dark = (10, 25, 30)`), and `netinspect.compose` composites similar shapes onto
real net. If the model learned that prototype rather than "damage", then its
false alarms should be regions that *match the prototype* — dark, elongated,
high-contrast against their surroundings — and bag3 should contain more such
regions than the clips it does not fire on.

Two predictions follow, and they are separable:

* **P1 — the false alarms look like the training damage.** Per-box statistics
  (darkness relative to surround, elongation, edge density) for false positives
  should sit close to those of the labelled damage boxes, not to random patches.
* **P2 — bag3 contains more prototype-matching structure than the other clips.**
  If P1 holds but P2 fails, the model is firing on something bag3 does not
  distinctively have, and the "mooring cords" story is wrong.

If the false alarms are *brighter* than their surroundings, the stated mechanism
is refuted outright: the training damage is dark, so a bright cord cannot be
matching it directly — the model would have to be firing on the shadow beside
the cord, or on something else entirely.

    python scripts/analyse_false_alarms.py

Writes per-box statistics, a comparison table, and a contact sheet of the
strongest false alarms so the numbers can be checked against the pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect.inference import NetInspector
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()
REPO = _common.REPO_ROOT

CLIPS = {
    "bag1": "data/processed/solaqua_frames",
    "bag2": "data/processed/solaqua_bag2",
    "bag3": "data/processed/solaqua_bag3",
    "diffday": "data/processed/solaqua_diffday",
}


def _luma(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def _box_stats(img: np.ndarray, x1, y1, x2, y2) -> dict | None:
    """Describe one box the way the synthetic generator describes damage.

    ``contrast`` is signed on purpose: the training damage is DARK, so a
    positive value (region darker than its surround) is consistent with the
    prototype and a negative one refutes it for that box.
    """
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    lum = _luma(img)
    inside = lum[y1:y2, x1:x2]

    # Surround = a ring around the box, so "contrast" is local rather than
    # against the whole frame, whose brightness varies with depth and lighting.
    pad_x, pad_y = max(4, (x2 - x1) // 2), max(4, (y2 - y1) // 2)
    sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    sx2, sy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    ring = lum[sy1:sy2, sx1:sx2].copy()
    ring[y1 - sy1:y2 - sy1, x1 - sx1:x2 - sx1] = np.nan
    surround = float(np.nanmean(ring)) if np.isfinite(ring).any() else float(lum.mean())

    bw, bh = x2 - x1, y2 - y1
    gx = np.abs(np.diff(inside, axis=1)).mean() if bw > 1 else 0.0
    gy = np.abs(np.diff(inside, axis=0)).mean() if bh > 1 else 0.0

    return {
        "mean_luma": float(inside.mean()),
        "surround_luma": surround,
        # >0 means DARKER than surround, i.e. consistent with the training damage.
        "contrast": float(surround - inside.mean()),
        "elongation": float(max(bw, bh) / max(1, min(bw, bh))),
        "area_frac": float((bw * bh) / (w * h)),
        "edge_density": float((gx + gy) / 2.0),
    }


def _summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": round(float(v.mean()), 3),
                  "median": round(float(np.median(v)), 3),
                  "p10": round(float(np.percentile(v, 10)), 3),
                  "p90": round(float(np.percentile(v, 90)), 3)}
    return out


def _contact_sheet(crops, out_path: Path, cols: int = 6, cell: int = 128) -> None:
    """A grid of the strongest false alarms — numbers should survive being looked at."""
    if not crops:
        return
    try:
        from PIL import Image
    except Exception:                                     # pragma: no cover
        return
    rows = (len(crops) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 20, 24))
    for i, arr in enumerate(crops):
        im = Image.fromarray(arr).resize((cell - 4, cell - 4))
        sheet.paste(im, ((i % cols) * cell + 2, (i // cols) * cell + 2))
    sheet.save(out_path)
    LOGGER.info("wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--method", default="yolo")
    ap.add_argument("--labelled", default="data/processed/real_composite",
                    help="labelled damage, for the prototype the model was trained on")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/results/false_alarm_anatomy")
    args = ap.parse_args()

    insp = NetInspector(yolo_weights=args.weights)
    out_dir = ensure_dir(args.out)

    # --- false alarms, per clip ------------------------------------------- #
    per_clip, fp_rows, strongest = {}, [], []
    for clip, rel in CLIPS.items():
        paths = list_images(REPO / rel)
        rows, alarming = [], 0
        for p in paths:
            img = read_image(p)
            boxes = insp.predict(img, method=args.method, conf=args.conf).boxes
            if boxes:
                alarming += 1
            for b in boxes:
                st = _box_stats(img, b.x1, b.y1, b.x2, b.y2)
                if st is None:
                    continue
                st.update(clip=clip, score=float(b.score), frame=p.name)
                rows.append(st)
                if clip == "bag3":
                    crop = img[max(0, int(b.y1)):int(b.y2), max(0, int(b.x1)):int(b.x2)]
                    if crop.size:
                        strongest.append((float(b.score), crop))
        per_clip[clip] = {"frames": len(paths), "alarming": alarming,
                          "rate": alarming / max(1, len(paths)),
                          "boxes": len(rows), "stats": _summarise(rows)}
        fp_rows += rows
        LOGGER.info("%s: %d/%d frames alarmed, %d boxes", clip, alarming, len(paths), len(rows))

    # --- the damage the model was actually trained on ---------------------- #
    from netinspect import dataset as D
    gt_rows = []
    for split in ("train", "test"):
        for s in D.load_dataset(Path(args.labelled) / "images" / split, fmt="yolo"):
            img = read_image(s.image)
            h, w = img.shape[:2]
            for b in s.boxes:
                st = _box_stats(img, b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h)
                if st:
                    gt_rows.append(st)

    fp = _summarise(fp_rows)
    gt = _summarise(gt_rows)

    print(f"\n{fp['n']} false-alarm boxes on real undamaged net "
          f"vs {gt['n']} labelled damage boxes\n")
    print(f"{'feature':16s} {'false alarms':>16s} {'trained-on damage':>20s}")
    for k in ("contrast", "mean_luma", "elongation", "edge_density", "area_frac"):
        print(f"{k:16s} {fp[k]['median']:>16.2f} {gt[k]['median']:>20.2f}")

    darker = sum(1 for r in fp_rows if r["contrast"] > 0)
    print("\nP1 — do the false alarms match the DARK training prototype?")
    print(f"     {darker}/{len(fp_rows)} false-alarm boxes are darker than their "
          f"surround ({darker / max(1, len(fp_rows)):.0%}).")
    verdict_p1 = ("CONSISTENT: the false alarms are dark regions, like the training damage"
                  if darker > 0.6 * len(fp_rows) else
                  "REFUTED: the false alarms are mostly BRIGHTER than their surround, so "
                  "they cannot be matching the dark synthetic damage directly")
    print(f"     {verdict_p1}")

    print("\nP2 — is bag3 distinctive?")
    for clip, d in per_clip.items():
        s = d["stats"]
        med = f"{s['contrast']['median']:.2f}" if s["n"] else "  n/a"
        print(f"     {clip:9s} {d['rate']:>6.1%} of frames, {d['boxes']:>4d} boxes, "
              f"median contrast {med}")

    _contact_sheet([c for _, c in sorted(strongest, key=lambda t: -t[0])[:24]],
                   Path(out_dir) / "bag3_false_alarms.png")

    write_json({"config": vars(args), "per_clip": per_clip,
                "false_alarms": fp, "trained_on_damage": gt,
                "fraction_darker_than_surround": darker / max(1, len(fp_rows)),
                "P1": verdict_p1},
               Path(out_dir) / "false_alarm_anatomy.json")
    print(f"\nwrote {out_dir}/false_alarm_anatomy.json")


if __name__ == "__main__":
    main()

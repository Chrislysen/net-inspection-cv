"""Per-pass inspection validity report — was this ROV run worth trusting?

A detector answers "is there damage in this frame". An operator needs a
different answer: **can I sign off this inspection, and what did it actually
cover?** A pass that produced no detections is only reassuring if the footage
was fit to judge in the first place.

This produces that report for one ROV pass, from the frame-to-telemetry join
already computed by ``scripts/analyze_operating_envelope.py``:

* **Coverage** — what fraction of frames were captured with a locked net-plane
  estimate inside the conditions the models were characterised in.
* **Per-frame verdict** — ``in_envelope`` / ``out_of_envelope`` / ``unknown``,
  each with a human-readable reason.
* **Detections** — raw per-frame counts *and* temporally confirmed events, since
  a defect that appears in one frame and never again is usually a flicker.
* **What was not inspected, and why** — the section that makes the report honest
  rather than reassuring.

Regulatory context: ``akvakulturdriftsforskriften §41`` already requires net
inspections to be journalled, and §37 requires nets be checked *"regelmessig
under driften"* without defining an interval. A machine-generated validity
record has a pre-existing home in that journal.

Honesty
-------
The envelope here describes **the range of conditions present in the evaluated
data**, not certified operating limits, and this repo's own analysis found that
standoff distance does *not* drive false alarms — scene identity does. So an
``in_envelope`` verdict means "these capture conditions resemble the evaluated
ones", never "detection is accurate here". Recall on real damage is unmeasured.

Examples
--------
    python scripts/inspection_report.py --clip 2024-08-22_14-29-05
    python scripts/inspection_report.py --all --model seg_gpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect.envelope import IN_ENVELOPE, OUT_OF_ENVELOPE, UNKNOWN, EnvelopeGate, EnvelopeSpec
from netinspect.utils import ensure_dir, get_logger, write_json

LOGGER = get_logger()

FRAME_CONDITIONS = "reports/results/operating_envelope/frame_conditions.parquet"

# The range of capture conditions actually present in the evaluated data.
# NOT certified operating limits — see the module docstring.
EVALUATED_ENVELOPE = EnvelopeSpec(
    standoff_min_m=0.19,
    standoff_max_m=1.54,
    speed_max_ms=0.35,
    require_lock=True,
    model="observed-data-range",
    evidence={
        "source": "reports/results/operating_envelope/operating_envelope.json",
        "meaning": ("Range of conditions present in the 638 evaluated frames. "
                    "Indicates similarity to the evaluated sample, not accuracy."),
    },
)


def build_report(df, clip: str, model: str, gate: EnvelopeGate) -> dict:
    """Assemble the validity report for one clip."""
    import numpy as np

    g = df[df["clip"] == clip].sort_values("msg_index").reset_index(drop=True)
    if g.empty:
        return {"clip": clip, "error": "no frames for this clip"}

    verdicts = [
        gate.check(
            standoff_m=(None if np.isnan(r.standoff) else float(r.standoff)),
            speed_ms=(None if np.isnan(r.net_speed) else float(r.net_speed)),
            locked=bool(r.locked),
        )
        for r in g.itertuples()
    ]
    summary = gate.summarise(verdicts)

    det_col = f"ndet_{model}"
    fp_col = f"fp_{model}"
    detections = None
    if det_col in g.columns:
        counts = g[det_col].fillna(0).astype(int).tolist()
        flags = g[fp_col].fillna(0).astype(int).tolist()

        # Temporal confirmation: a run of consecutive frames with detections.
        runs, run = [], 0
        for c in counts:
            if c > 0:
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 0
        if run:
            runs.append(run)
        confirmed = [r for r in runs if r >= 3]

        trusted = [f for f, v in zip(flags, verdicts) if v.status == IN_ENVELOPE]
        untrusted = [f for f, v in zip(flags, verdicts) if v.status != IN_ENVELOPE]

        detections = {
            "model": model,
            "frames_with_detection": int(sum(1 for c in counts if c > 0)),
            "total_detections": int(sum(counts)),
            "candidate_events": len(runs),
            "confirmed_events_min3_frames": len(confirmed),
            "longest_event_frames": max(runs) if runs else 0,
            "detection_rate_in_envelope": (
                round(sum(trusted) / len(trusted), 4) if trusted else None),
            "detection_rate_outside_envelope": (
                round(sum(untrusted) / len(untrusted), 4) if untrusted else None),
            "note": ("These frames show UNDAMAGED net, so every detection here is "
                     "a false alarm. On a real pass the same fields would be "
                     "candidate defects requiring human review."),
        }

    # Capture QUALITY, fitted from the data rather than assumed.
    #
    # Geometry alone is not enough and this repo's own numbers say so: clip
    # 14-29-05 passes every standoff/speed/lock check and still false-alarms on
    # a third of its frames. The variable that actually discriminates is
    # sharpness (r = +0.60 for the detector), so the report checks it too and
    # reports the two dimensions separately rather than blending them.
    quality = None
    if "sharpness" in g.columns and fp_col in g.columns:
        from netinspect.image_quality import fit_band

        band = fit_band(df["sharpness"].to_numpy(), df[fp_col].fillna(0).to_numpy(int),
                        target_rate=0.05, model=model, metric="sharpness")
        if band.evidence and band.evidence.get("fitted"):
            lo, hi = band.sharpness_min, band.sharpness_max
            vals = g["sharpness"].to_numpy()
            inside = [(lo is None or v >= lo) and (hi is None or v <= hi) for v in vals]
            quality = {
                "metric": "sharpness",
                "validated_band": [lo, hi],
                "clip_mean": round(float(vals.mean()), 1),
                "frames_in_band": int(sum(inside)),
                "share_in_band": round(sum(inside) / len(inside), 4),
                "fitted_on_frames": band.evidence.get("frames"),
                "meaning": ("Sharpness range in which this model's measured "
                            "false-alarm rate stayed under 5% with 95% confidence. "
                            "Geometry and quality are separate checks: a pass can "
                            "satisfy every standoff and speed criterion and still "
                            "sit outside the quality band."),
            }
        else:
            quality = {"metric": "sharpness", "fitted": False,
                       "note": band.evidence.get("note") if band.evidence else None}

    not_inspected = []
    for status, label in ((OUT_OF_ENVELOPE, "captured outside evaluated conditions"),
                          (UNKNOWN, "capture conditions could not be verified")):
        n = summary["counts"].get(status, 0)
        if n:
            reasons: dict[str, int] = {}
            for v in verdicts:
                if v.status == status:
                    for r in (v.reasons or ["unspecified"]):
                        reasons[r] = reasons.get(r, 0) + 1
            not_inspected.append({
                "status": status, "frames": n,
                "share": round(n / len(verdicts), 4),
                "meaning": label,
                "top_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:3]),
            })

    return {
        "clip": clip,
        "day": clip[:10],
        "frames": int(len(g)),
        "capture": {
            "standoff_m_mean": round(float(g["standoff"].mean()), 3),
            "standoff_m_range": [round(float(g["standoff"].min()), 2),
                                 round(float(g["standoff"].max()), 2)],
            "net_speed_ms_mean": round(float(g["net_speed"].mean()), 3),
            "depth_m_mean": round(float(g["depth"].mean()), 2)
            if "depth" in g else None,
            "water_temperature_c": round(float(g["temperature"].mean()), 2)
            if "temperature" in g else None,
            "sharpness_mean": round(float(g["sharpness"].mean()), 1)
            if "sharpness" in g else None,
        },
        "validity_geometry": summary,
        "validity_quality": quality,
        "detections": detections,
        "not_inspected": not_inspected,
        "caveats": [
            "'in_envelope' means capture conditions resemble the evaluated sample. "
            "It does NOT mean detection is accurate under those conditions.",
            "Recall on real damage has never been measured; all damage data in this "
            "project is synthetic composited onto real undamaged net.",
            "This repo's own analysis rejected standoff distance as a driver of "
            "false alarms — scene identity dominates. Treat the envelope as a "
            "similarity check, not a performance guarantee.",
        ],
    }


def print_report(rep: dict) -> None:
    if "error" in rep:
        print(f"  {rep['clip']}: {rep['error']}")
        return
    v, c = rep["validity_geometry"], rep["capture"]
    print("\n" + "=" * 78)
    print(f"INSPECTION PASS REPORT — {rep['clip']}")
    print("=" * 78)
    print(f"  frames            {rep['frames']}")
    print(f"  standoff          {c['standoff_m_mean']} m "
          f"(range {c['standoff_m_range'][0]}–{c['standoff_m_range'][1]})")
    print(f"  sweep speed       {c['net_speed_ms_mean']} m/s")
    if c.get("depth_m_mean") is not None:
        print(f"  depth / temp      {c['depth_m_mean']} m · {c['water_temperature_c']} degC")
    print()
    print(f"  VERDICT           {v['verdict']}")
    print(f"  coverage          {v['compliance']:.1%} of frames verified in-envelope")
    print(f"  breakdown         {v['counts']}")
    print(f"  envelope          {v['spec_human']}")

    q = rep.get("validity_quality")
    if q and q.get("validated_band"):
        lo, hi = q["validated_band"]
        rng = f"{lo if lo is not None else 'any'}-{hi if hi is not None else 'any'}"
        print()
        print(f"  capture quality   sharpness {q['clip_mean']} "
              f"(validated band {rng})")
        print(f"                    {q['share_in_band']:.1%} of frames inside the band")

    d = rep.get("detections")
    if d:
        print()
        print(f"  detections ({d['model']})")
        print(f"    frames with a detection   {d['frames_with_detection']}")
        print(f"    candidate events          {d['candidate_events']}")
        print(f"    confirmed (>=3 frames)    {d['confirmed_events_min3_frames']}")
        if d["detection_rate_in_envelope"] is not None:
            print(f"    rate inside envelope      {d['detection_rate_in_envelope']:.3f}")
        if d["detection_rate_outside_envelope"] is not None:
            print(f"    rate outside envelope     {d['detection_rate_outside_envelope']:.3f}")

    if rep["not_inspected"]:
        print()
        print("  NOT RELIABLY INSPECTED")
        for n in rep["not_inspected"]:
            print(f"    {n['frames']:4d} frames ({n['share']:.1%}) — {n['meaning']}")
            for reason, cnt in n["top_reasons"].items():
                print(f"          {cnt:4d}x {reason}")
    else:
        print("\n  NOT RELIABLY INSPECTED: none — every frame verified.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None, help="Clip id, e.g. 2024-08-22_14-29-05")
    ap.add_argument("--all", action="store_true", help="Report on every clip")
    ap.add_argument("--model", default="det_v1",
                    help="Detection model column to summarise")
    ap.add_argument("--conditions", default=FRAME_CONDITIONS)
    ap.add_argument("--out", default="reports/results/inspection_reports")
    args = ap.parse_args()

    import pandas as pd

    path = Path(args.conditions)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/analyze_operating_envelope.py first.")
    df = pd.read_parquet(path)

    clips = sorted(df["clip"].unique()) if args.all else ([args.clip] if args.clip else [])
    if not clips:
        ap.error("Provide --clip or --all. Available: "
                 + ", ".join(sorted(df["clip"].unique())))

    gate = EnvelopeGate(EVALUATED_ENVELOPE)
    reports = [build_report(df, clip, args.model, gate) for clip in clips]
    for rep in reports:
        print_report(rep)

    out_dir = ensure_dir(args.out)
    for rep in reports:
        if "error" not in rep:
            write_json(rep, out_dir / f"{rep['clip']}_inspection_report.json")

    ok = [r for r in reports if "error" not in r]
    if len(ok) > 1:
        print("\n" + "=" * 78)
        print("ACROSS PASSES")
        print("=" * 78)
        for r in ok:
            q = r.get("validity_quality") or {}
            qs = (f"{q['share_in_band']:6.1%}" if q.get("share_in_band") is not None
                  else "     -")
            print(f"  {r['clip']:22s} geometry {r['validity_geometry']['compliance']:6.1%}"
                  f"   quality {qs}   det {r['detections']['frames_with_detection'] if r.get('detections') else 0:3d}")
        print("\n  What this table shows, stated plainly: neither check predicts")
        print("  which pass breaks the detector. 14-29-05 clears every geometric")
        print("  criterion and still false-alarms on 61 frames, while 14-47-39 sits")
        print("  outside the same fitted quality band and produces none. The clip")
        print("  effect is not reducible to either measured covariate — which is")
        print("  the finding, not a defect in the report. Capture checks bound when")
        print("  a result is comparable to the evaluated sample; they do not")
        print("  substitute for scene-level validation on real labelled damage.")

    print(f"\nWrote {len(ok)} report(s) to {out_dir}")
    print("\nCaveat: 'in_envelope' is a similarity check against the evaluated "
          "sample,\nnot a performance guarantee. Recall on real damage is unmeasured.")


if __name__ == "__main__":
    main()

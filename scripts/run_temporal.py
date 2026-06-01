"""Run a method over a frame SEQUENCE with temporal confirmation.

Compares raw per-frame detections against temporally-confirmed ones (detections
that persist over several frames). On real undamaged footage this quantifies how
much transient-clutter false alarm the temporal filter removes.

Frames are processed in filename order (so an extracted, CONTIGUOUS sequence is
required — e.g. `fetch_solaqua.py ... --every-n 1`).

Example
-------
    python scripts/run_temporal.py --method classical --source data/processed/solaqua_seq \\
        --out outputs/temporal --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.temporal import TemporalConfig, Tracker
from netinspect.utils import (
    ensure_dir,
    get_logger,
    list_images,
    read_image,
    write_image,
    write_json,
)
from netinspect.visualize import overlay_boxes, side_by_side

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="classical", choices=["classical", "yolo", "patchcore"])
    ap.add_argument("--source", required=True, help="Directory of CONTIGUOUS frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--anomaly-model", default=None)
    ap.add_argument("--patchcore-model", default=None)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=2)
    ap.add_argument("--iou-match", type=float, default=0.2)
    ap.add_argument("--undamaged", action="store_true", default=True,
                    help="Treat all detections as false alarms (default for SOLAQUA)")
    args = ap.parse_args()

    ccfg = ClassicalConfig()
    if args.config:
        params = _common.load_yaml(args.config).get("classical", {})
        ccfg = ClassicalConfig(**{k: v for k, v in params.items() if k in ClassicalConfig().__dict__})

    insp = NetInspector(classical_cfg=ccfg, patchcore_model_path=args.patchcore_model,
                        yolo_weights=args.yolo_weights)
    if args.method not in insp.available_methods():
        print(f"Method '{args.method}' unavailable ({insp.available_methods()}).")
        return

    frames = list_images(args.source)
    if not frames:
        print(f"No frames in {args.source}.")
        return

    tracker = Tracker(TemporalConfig(iou_match=args.iou_match, min_hits=args.min_hits,
                                     max_age=args.max_age, conf=args.conf))
    out = ensure_dir(args.out)
    overlay_dir = ensure_dir(out / "overlays")

    raw_total = conf_total = raw_fp_frames = conf_fp_frames = 0
    for path in frames:
        img = read_image(path)
        raw = insp.predict(img, method=args.method, conf=args.conf).boxes
        confirmed = tracker.update(raw)
        raw_total += len(raw)
        conf_total += len(confirmed)
        raw_fp_frames += int(len(raw) > 0)
        conf_fp_frames += int(len(confirmed) > 0)
        write_image(overlay_dir / f"{path.stem}.jpg",
                    side_by_side(overlay_boxes(img, preds=raw),
                                 overlay_boxes(img, preds=confirmed)))

    n = len(frames)
    reduction = 1 - (conf_total / raw_total) if raw_total else 0.0
    summary = {
        "method": args.method, "frames": n, "undamaged": args.undamaged,
        "temporal": {"min_hits": args.min_hits, "max_age": args.max_age, "iou_match": args.iou_match},
        "raw_detections": raw_total,
        "confirmed_detections": conf_total,
        "detection_reduction": round(reduction, 3),
        "raw_fp_frame_rate": round(raw_fp_frames / n, 3),
        "confirmed_fp_frame_rate": round(conf_fp_frames / n, 3),
    }
    write_json(summary, out / "temporal.json")
    print(f"Frames: {n}  (left=raw, right=temporally-confirmed in overlays)")
    print(f"Raw detections:        {raw_total}  (FP frame rate {summary['raw_fp_frame_rate']:.0%})")
    print(f"Confirmed detections:  {conf_total}  (FP frame rate {summary['confirmed_fp_frame_rate']:.0%})")
    if args.undamaged:
        print(f"=> Temporal confirmation removed {summary['detection_reduction']:.0%} of false alarms "
              f"on real undamaged footage.")
    print(f"Wrote {out / 'temporal.json'}")


if __name__ == "__main__":
    main()

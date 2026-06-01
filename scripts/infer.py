"""Unified batch / video inference over any method (classical | anomaly | yolo).

Consolidates the per-method scripts behind one CLI using netinspect.inference.
Writes overlays, a JSONL of per-frame detections, and a run manifest for
reproducibility.

Examples
--------
    python scripts/infer.py --method classical --source data/processed/solaqua_frames --out outputs/infer
    python scripts/infer.py --method yolo --yolo-weights runs/detect/train/weights/best.pt \\
        --source data/raw/solaqua/clip.bag --out outputs/infer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.utils import (VIDEO_EXTENSIONS, ensure_dir, get_logger,
                              list_images, read_image, write_image, write_json)
from netinspect.visualize import overlay_boxes

LOGGER = get_logger()


def _iter_frames(source: Path, every_n: int, max_frames: int | None):
    """Yield (name, rgb) from an image dir, a video, or a ROS bag."""
    if source.is_dir():
        for p in list_images(source):
            yield p.stem, read_image(p)
        return
    suffix = source.suffix.lower()
    if suffix == ".bag":
        from netinspect.solaqua import extract_bag_frames
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        for p in extract_bag_frames(source, tmp, every_n=every_n, max_frames=max_frames):
            yield p.stem, read_image(p)
        return
    if suffix in VIDEO_EXTENSIONS:
        from netinspect.utils import optional_import
        cv2 = optional_import("cv2")
        cap = cv2.VideoCapture(str(source))
        idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                yield f"{source.stem}_{idx:06d}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                saved += 1
                if max_frames and saved >= max_frames:
                    break
            idx += 1
        cap.release()
        return
    raise ValueError(f"Unsupported source: {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="classical", choices=["classical", "anomaly", "yolo"])
    ap.add_argument("--source", required=True, help="image dir, video, or .bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="classical config yaml")
    ap.add_argument("--anomaly-model", default=None)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every-n", type=int, default=10, help="video/bag frame stride")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    ccfg = ClassicalConfig()
    if args.config:
        params = _common.load_yaml(args.config).get("classical", {})
        ccfg = ClassicalConfig(**{k: v for k, v in params.items() if k in ClassicalConfig().__dict__})

    inspector = NetInspector(classical_cfg=ccfg, anomaly_model_path=args.anomaly_model,
                             yolo_weights=args.yolo_weights)
    if args.method not in inspector.available_methods():
        print(f"Method '{args.method}' unavailable. Available: {inspector.available_methods()}")
        print("(Provide --anomaly-model / --yolo-weights, or install ultralytics.)")
        return

    out = ensure_dir(args.out)
    overlay_dir = ensure_dir(out / "overlays")
    jsonl_path = out / "detections.jsonl"
    n = total = 0
    times = []
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for name, img in _iter_frames(Path(args.source), args.every_n, args.max_frames):
            res = inspector.predict(img, method=args.method, conf=args.conf)
            rec = {"frame": name, **res.to_dict()}
            jf.write(json.dumps(rec) + "\n")
            if not args.no_overlays:
                vis = res.heatmap if res.heatmap is not None else overlay_boxes(img, preds=res.boxes)
                write_image(overlay_dir / f"{name}.jpg", vis)
            n += 1
            total += len(res.boxes)
            times.append(res.elapsed_ms)

    manifest = {
        "method": args.method, "source": str(args.source), "conf": args.conf,
        "frames": n, "total_detections": total,
        "mean_detections_per_frame": round(total / n, 3) if n else 0,
        "mean_latency_ms": round(sum(times) / len(times), 2) if times else 0,
        "classical_config": ccfg.__dict__ if args.method == "classical" else None,
    }
    write_json(manifest, out / "run_manifest.json")
    print(f"Processed {n} frames with '{args.method}': {total} detections, "
          f"mean {manifest['mean_latency_ms']} ms/frame.")
    print(f"Wrote {jsonl_path} and run_manifest.json")


if __name__ == "__main__":
    main()

"""Train a YOLOv8 detection/segmentation baseline on a YOLO-format dataset.

Requires `ultralytics` (and real labelled data). If ultralytics is missing or
the dataset config is absent, this prints clear guidance instead of crashing.

Examples
--------
    python scripts/train_yolo.py --data configs/yolo_dataset.yaml --epochs 50
    python scripts/train_yolo.py --data configs/yolo_dataset.yaml --task segment --model yolov8n-seg.pt
"""
from __future__ import annotations

import argparse
import sys

import _common  # noqa: F401

from netinspect.model_baseline import ULTRALYTICS_HINT, YoloConfig, train, ultralytics_available


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="YOLO dataset YAML")
    ap.add_argument("--task", default="detect", choices=["detect", "segment"])
    ap.add_argument("--model", default="yolov8n.pt", help="Base weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda device or 'cpu'")
    ap.add_argument("--patience", type=int, default=None, help="Early-stopping patience")
    # Photometric/geometric augmentation overrides (None -> ultralytics default).
    # Stronger HSV augmentation simulates day-to-day lighting/turbidity shift, the
    # dominant out-of-distribution failure mode for underwater net inspection.
    ap.add_argument("--hsv-h", type=float, default=None, help="Hue jitter (water colour cast)")
    ap.add_argument("--hsv-s", type=float, default=None, help="Saturation jitter")
    ap.add_argument("--hsv-v", type=float, default=None, help="Value/brightness jitter (turbidity)")
    ap.add_argument("--degrees", type=float, default=None, help="Rotation degrees")
    ap.add_argument("--translate", type=float, default=None, help="Translation fraction")
    ap.add_argument("--scale", type=float, default=None, help="Scale jitter")
    ap.add_argument("--perspective", type=float, default=None, help="Perspective warp")
    ap.add_argument("--flipud", type=float, default=None, help="Vertical-flip probability")
    args = ap.parse_args()

    augment = {k: getattr(args, k) for k in
               ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "perspective", "flipud")}
    augment = {k: v for k, v in augment.items() if v is not None} or None

    if not ultralytics_available():
        print("ultralytics is not installed, so training cannot run.")
        print(ULTRALYTICS_HINT)
        print("\nThe code path is in place: install ultralytics and supply a YOLO "
              "dataset to train. See configs/yolo_dataset.yaml and the README.")
        sys.exit(1)

    cfg = YoloConfig(task=args.task, model=args.model, imgsz=args.imgsz,
                     epochs=args.epochs, batch=args.batch, device=args.device,
                     augment=augment, patience=args.patience)
    best = train(args.data, cfg)
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()

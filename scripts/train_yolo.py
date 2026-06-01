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
    args = ap.parse_args()

    if not ultralytics_available():
        print("ultralytics is not installed, so training cannot run.")
        print(ULTRALYTICS_HINT)
        print("\nThe code path is in place: install ultralytics and supply a YOLO "
              "dataset to train. See configs/yolo_dataset.yaml and the README.")
        sys.exit(1)

    cfg = YoloConfig(task=args.task, model=args.model, imgsz=args.imgsz,
                     epochs=args.epochs, batch=args.batch, device=args.device)
    best = train(args.data, cfg)
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()

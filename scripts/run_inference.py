"""Run YOLO inference over a directory of images using trained weights.

Writes predictions JSON (same schema as the classical baseline) and overlays,
so the evaluator and visualiser treat both methods identically.

Examples
--------
    python scripts/run_inference.py --images data/processed/images --weights runs/train/weights/best.pt --out outputs/predictions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common  # noqa: F401

from netinspect.model_baseline import (
    ULTRALYTICS_HINT,
    YoloConfig,
    load_model,
    predict_image,
    ultralytics_available,
)
from netinspect.utils import (
    ensure_dir,
    get_logger,
    list_images,
    read_image,
    save_predictions,
    write_image,
)
from netinspect.visualize import overlay_boxes

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True, help="Trained YOLO .pt checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="detect", choices=["detect", "segment"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    if not ultralytics_available():
        print("ultralytics is not installed, so YOLO inference cannot run.")
        print(ULTRALYTICS_HINT)
        sys.exit(1)

    images = list_images(args.images)
    if not images:
        print(f"No images found in {args.images}.")
        return

    cfg = YoloConfig(task=args.task, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    model = load_model(args.weights, task=args.task)

    out = Path(args.out)
    overlay_dir = ensure_dir(out / "overlays")
    preds_by_image: dict[str, list] = {}
    for path in images:
        img = read_image(path)
        boxes = predict_image(model, img, cfg)
        preds_by_image[path.name] = boxes
        if not args.no_overlays:
            write_image(overlay_dir / f"{path.stem}_overlay.jpg",
                        overlay_boxes(img, preds=boxes))
        LOGGER.info("%s: %d detection(s)", path.name, len(boxes))

    save_predictions(preds_by_image, out / "preds.json",
                     meta={"method": "yolo", "weights": str(args.weights),
                           "conf": args.conf, "iou": args.iou})
    print(f"\nWrote predictions to {out / 'preds.json'}")
    if not args.no_overlays:
        print(f"Overlays: {overlay_dir}")


if __name__ == "__main__":
    main()

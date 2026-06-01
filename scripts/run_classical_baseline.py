"""Run the classical (OpenCV) baseline over a directory of images.

Writes predictions (JSON + CSV) and overlay visualisations.

Examples
--------
    python scripts/run_classical_baseline.py --images data/processed/images --out outputs/classical
    python scripts/run_classical_baseline.py --images data/sample/images --out outputs/classical --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from netinspect.classical_baseline import ClassicalConfig, detect
from netinspect.utils import (ensure_dir, get_logger, list_images, read_image,
                              save_predictions, write_image)
from netinspect.visualize import overlay_boxes

LOGGER = get_logger()


def _config_from_yaml(path: str | None) -> ClassicalConfig:
    if not path:
        return ClassicalConfig()
    cfg = _common.load_yaml(path).get("classical", {})
    known = ClassicalConfig().__dict__
    return ClassicalConfig(**{k: v for k, v in cfg.items() if k in known})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Directory of images")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--config", default=None, help="YAML config (configs/baseline.yaml)")
    ap.add_argument("--no-overlays", action="store_true", help="Skip overlay images")
    args = ap.parse_args()

    cfg = _config_from_yaml(args.config)
    images = list_images(args.images)
    if not images:
        print(f"No images found in {args.images}.")
        return

    out = Path(args.out)
    overlay_dir = ensure_dir(out / "overlays")
    preds_by_image: dict[str, list] = {}
    csv_rows = ["image,x1,y1,x2,y2,score,class_name"]

    for path in images:
        img = read_image(path)
        result = detect(img, cfg)
        preds_by_image[path.name] = result.boxes
        for b in result.boxes:
            csv_rows.append(f"{path.name},{b.x1:.1f},{b.y1:.1f},{b.x2:.1f},"
                            f"{b.y2:.1f},{b.score:.4f},{b.class_name}")
        if not args.no_overlays:
            write_image(overlay_dir / f"{path.stem}_overlay.jpg",
                        overlay_boxes(img, preds=result.boxes))
        LOGGER.info("%s: %d candidate region(s)", path.name, len(result.boxes))

    save_predictions(preds_by_image, out / "preds.json",
                     meta={"method": "classical_baseline", "config": cfg.__dict__})
    (out / "preds.csv").write_text("\n".join(csv_rows), encoding="utf-8")

    n_boxes = sum(len(v) for v in preds_by_image.values())
    print(f"\nProcessed {len(images)} images, {n_boxes} candidate regions.")
    print(f"Predictions: {out / 'preds.json'} / preds.csv")
    if not args.no_overlays:
        print(f"Overlays:    {overlay_dir}")
    print("\nNOTE: the classical baseline flags visual anomalies for review. "
          "It is a sanity-check, not a validated damage detector.")


if __name__ == "__main__":
    main()

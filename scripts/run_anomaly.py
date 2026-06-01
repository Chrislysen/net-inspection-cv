"""Score frames with a trained anomaly model; save heatmaps, boxes, predictions.

Example
-------
    python scripts/run_anomaly.py --images data/processed/solaqua_frames \\
        --model outputs/anomaly/model --out outputs/anomaly
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from netinspect.anomaly import AnomalyModel, anomaly_heatmap, score_image
from netinspect.utils import (ensure_dir, get_logger, list_images, read_image,
                              save_predictions, write_image, write_json)
from netinspect.visualize import overlay_boxes, side_by_side, write_gallery_markdown

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True, help="Model prefix from train_anomaly.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AnomalyModel.load(args.model)
    images = list_images(args.images)
    if not images:
        print(f"No images in {args.images}.")
        return

    out = ensure_dir(args.out)
    heat_dir = ensure_dir(out / "heatmaps")
    preds_by_image: dict[str, list] = {}
    per_image = []
    for path in images:
        img = read_image(path)
        res = score_image(img, model)
        preds_by_image[path.name] = res.boxes
        per_image.append({"image": path.name, "num_regions": len(res.boxes),
                          "max_score": round(res.max_score, 3)})
        heat = anomaly_heatmap(img, res, model)
        boxed = overlay_boxes(img, preds=res.boxes)
        write_image(heat_dir / f"{path.stem}.jpg", side_by_side(boxed, heat))

    n = len(images)
    flagged = sum(1 for p in per_image if p["num_regions"] > 0)
    summary = {
        "method": "patch-feature Mahalanobis anomaly model",
        "note": "Flags deviation from normal net appearance, NOT validated damage. "
                "On undamaged SOLAQUA frames, flagged regions are out-of-distribution "
                "content (fish, biofouling, markers, lighting), i.e. candidate false alarms.",
        "threshold": model.threshold,
        "num_frames": n,
        "frames_flagged": flagged,
        "flag_rate": round(flagged / n, 3),
        "per_image": per_image,
    }
    write_json(summary, out / "anomaly.json")
    save_predictions(preds_by_image, out / "preds.json",
                     meta={"method": "anomaly_mahalanobis"})

    worst = sorted(per_image, key=lambda p: p["max_score"], reverse=True)[:8]
    entries = [{"image": f"heatmaps/{Path(p['image']).stem}.jpg",
                "caption": f"{p['image']}: max score {p['max_score']} "
                           f"({p['num_regions']} region(s))"} for p in worst]
    write_gallery_markdown(out, entries, "Anomaly model on real SOLAQUA frames (boxes | heatmap)")

    print(f"Scored {n} frames. Flagged {flagged}/{n} ({100*summary['flag_rate']:.0f}%).")
    print(f"Wrote {out / 'anomaly.json'}; heatmaps in {heat_dir}")
    print("\nReminder: anomaly = deviation from normal net, not confirmed damage.")


if __name__ == "__main__":
    main()

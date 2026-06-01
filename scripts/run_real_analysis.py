"""Robustness / false-positive analysis of the classical baseline on real frames.

SOLAQUA frames are real but **undamaged**, so any region the classical baseline
flags is effectively a **false alarm**. This script quantifies that false-alarm
behaviour — the honest real-data counterpart to the synthetic demo — and saves
overlays so you can see *what* triggers the heuristic (biofouling, fish, markers,
shadows, lighting).

Example
-------
    python scripts/run_real_analysis.py --images data/processed/solaqua_frames --out outputs/real_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from netinspect.classical_baseline import ClassicalConfig, detect
from netinspect.utils import (ensure_dir, get_logger, list_images, read_image,
                              save_predictions, write_image, write_json)
from netinspect.visualize import overlay_boxes, write_gallery_markdown

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Directory of real frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="configs/baseline.yaml")
    ap.add_argument("--assume-undamaged", action="store_true", default=True,
                    help="Treat every detection as a false positive (SOLAQUA default)")
    args = ap.parse_args()

    cfg = ClassicalConfig()
    if args.config:
        params = _common.load_yaml(args.config).get("classical", {})
        cfg = ClassicalConfig(**{k: v for k, v in params.items() if k in ClassicalConfig().__dict__})

    images = list_images(args.images)
    if not images:
        print(f"No images in {args.images}. Run scripts/fetch_solaqua.py first.")
        return

    out = ensure_dir(args.out)
    overlay_dir = ensure_dir(out / "overlays")
    preds_by_image: dict[str, list] = {}
    per_image = []
    for path in images:
        img = read_image(path)
        res = detect(img, cfg)
        preds_by_image[path.name] = res.boxes
        per_image.append({"image": path.name, "num_detections": len(res.boxes),
                          "max_score": max((b.score for b in res.boxes), default=0.0)})
        write_image(overlay_dir / f"{path.stem}.jpg", overlay_boxes(img, preds=res.boxes))

    n = len(images)
    total_det = sum(p["num_detections"] for p in per_image)
    frames_with_det = sum(1 for p in per_image if p["num_detections"] > 0)
    summary = {
        "dataset": "SOLAQUA (real, undamaged nets, no damage labels)",
        "interpretation": "Nets are undamaged, so all detections are candidate FALSE POSITIVES.",
        "num_frames": n,
        "total_detections": total_det,
        "mean_detections_per_frame": round(total_det / n, 3),
        "frames_with_any_detection": frames_with_det,
        "false_positive_frame_rate": round(frames_with_det / n, 3),
        "per_image": per_image,
    }
    write_json(summary, out / "real_analysis.json")
    save_predictions(preds_by_image, out / "preds.json",
                     meta={"method": "classical_baseline", "dataset": "SOLAQUA"})

    # Gallery of the frames with the most detections (most informative failures).
    worst = sorted(per_image, key=lambda p: p["num_detections"], reverse=True)[:8]
    entries = [{"image": f"overlays/{Path(p['image']).stem}.jpg",
                "caption": f"{p['image']}: {p['num_detections']} false alarm(s)"} for p in worst]
    write_gallery_markdown(out, entries, "Classical baseline on real SOLAQUA frames (false positives)")

    print(f"Frames analysed:            {n}")
    print(f"Total detections (FP):      {total_det}")
    print(f"Mean detections per frame:  {summary['mean_detections_per_frame']}")
    print(f"Frames with >=1 false alarm: {frames_with_det}/{n} "
          f"({100*summary['false_positive_frame_rate']:.0f}%)")
    print(f"\nWrote {out / 'real_analysis.json'} and overlays in {overlay_dir}")
    print("\nThis is the honest real-data result: a simple darkness/texture heuristic "
          "raises frequent false alarms on biofouling, fish, markers and lighting. "
          "It motivates a learned model (and an agreed FP/FN trade-off).")


if __name__ == "__main__":
    main()

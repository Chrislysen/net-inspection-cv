"""Run the full demo pipeline and collect assets for the technical report.

Steps (all on the synthetic placeholder dataset unless --images is given):
  1. ensure a dataset exists (generate synthetic if needed),
  2. run the classical baseline,
  3. evaluate against ground truth,
  4. copy a few overlay + failure-case images into reports/assets,
  5. write reports/assets/metrics.json and a gallery.md.

This makes the report reproducible from a single command. Synthetic numbers are
for pipeline verification only and are labelled as such in the report.

Example
-------
    python scripts/make_report_assets.py
    python scripts/make_report_assets.py --images data/processed/images --labels data/processed/labels
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _common  # noqa: F401
from netinspect.classical_baseline import ClassicalConfig, detect
from netinspect.data import load_dataset, summarize_dataset
from netinspect.evaluate import (best_f1_threshold, confidence_sweep,
                                  evaluate_detection, evaluate_image_level)
from netinspect.synthetic import generate_dataset
from netinspect.utils import ensure_dir, read_image, write_image, write_json
from netinspect.visualize import (overlay_boxes, overlay_match,
                                  write_gallery_markdown)

REPO = _common.REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=None, help="Image dir (default: synthetic demo)")
    ap.add_argument("--labels", default=None, help="YOLO labels dir")
    ap.add_argument("--out", default=str(REPO / "reports" / "assets"))
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--max-gallery", type=int, default=6)
    args = ap.parse_args()

    assets = ensure_dir(args.out)
    synthetic = args.images is None

    # 1. Dataset.
    if synthetic:
        sample_root = REPO / "data" / "sample"
        if not (sample_root / "images").exists() or not list((sample_root / "images").glob("*")):
            print("Generating synthetic placeholder dataset...")
            generate_dataset(sample_root, n_images=8, seed=0)
        images_dir, labels_dir = sample_root / "images", sample_root / "labels"
    else:
        images_dir, labels_dir = Path(args.images), (Path(args.labels) if args.labels else None)

    samples = load_dataset(images_dir, labels_dir)
    if not samples:
        print(f"No images in {images_dir}; nothing to do.")
        return
    summary = summarize_dataset(samples)

    # 2. Classical baseline.
    cfg = ClassicalConfig()
    preds_by_image: dict[str, list] = {}
    overlays_dir = ensure_dir(assets / "overlays")
    for s in samples:
        img = read_image(s.image_path)
        res = detect(img, cfg)
        preds_by_image[s.image_path.name] = res.boxes
        write_image(overlays_dir / f"{s.stem}.jpg", overlay_boxes(img, preds=res.boxes, gts=s.boxes))

    # 3. Evaluate (if GT present).
    gts_by_image = {s.image_path.name: s.boxes for s in samples if s.label_path is not None}
    metrics = {"dataset_summary": summary, "synthetic": synthetic}
    gallery_entries = []

    if gts_by_image:
        det = evaluate_detection(preds_by_image, gts_by_image, args.iou)
        sweep = confidence_sweep(preds_by_image, gts_by_image, args.iou)
        metrics["detection"] = det["overall"]
        metrics["image_level"] = evaluate_image_level(preds_by_image, gts_by_image)
        metrics["confidence_sweep"] = sweep
        metrics["best_f1_threshold"] = best_f1_threshold(sweep)

        # 4. Failure-case overlays.
        failures_dir = ensure_dir(assets / "failures")
        sample_by_name = {s.image_path.name: s for s in samples}
        n_fail = 0
        for m in det["matches"]:
            if not (m.fp or m.fn):
                continue
            s = sample_by_name[m.image]
            img = read_image(s.image_path)
            vis = overlay_match(img, preds_by_image[m.image], s.boxes,
                                m.fp_pred_idx, m.fn_gt_idx)
            write_image(failures_dir / f"{s.stem}.jpg", vis)
            n_fail += 1
        metrics["num_failure_overlays"] = n_fail

    # 5. Gallery (overlays).
    for s in samples[:args.max_gallery]:
        rel = f"overlays/{s.stem}.jpg"
        gallery_entries.append({"image": rel,
                                "caption": f"{s.image_path.name} "
                                           f"({len(preds_by_image.get(s.image_path.name, []))} pred / "
                                           f"{len(s.boxes)} GT)"})
    write_gallery_markdown(assets, gallery_entries, "Classical baseline — demo gallery")
    write_json(metrics, assets / "metrics.json")

    print(f"Report assets written to {assets}")
    if "detection" in metrics:
        o = metrics["detection"]
        print(f"  Detection (synthetic): P={o['precision']:.3f} R={o['recall']:.3f} "
              f"F1={o['f1']:.3f} AP={o['ap']:.3f}")
    if synthetic:
        print("\n*** Metrics are on SYNTHETIC placeholder data — pipeline verification "
              "only, NOT real-world performance. ***")


if __name__ == "__main__":
    main()

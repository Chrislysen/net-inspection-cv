"""Evaluate predictions against YOLO ground-truth labels.

Computes detection metrics (precision/recall/F1/AP), an image-level
"contains damage?" score, a confidence sweep, and writes per-image
match overlays plus explicit false-positive / false-negative lists.

If labels are missing, it says so honestly and falls back to a qualitative
gallery rather than inventing numbers.

Examples
--------
    python scripts/evaluate.py --preds outputs/classical/preds.json \\
        --images data/processed/images --labels data/processed/labels --out outputs/eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from netinspect.data import load_dataset
from netinspect.evaluate import (best_f1_threshold, confidence_sweep,
                                  evaluate_detection, evaluate_image_level)
from netinspect.utils import (ensure_dir, list_images, load_predictions,
                              read_image, write_image, write_json)
from netinspect.visualize import overlay_match, write_gallery_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="Predictions JSON")
    ap.add_argument("--images", required=True,
                    help="Image directory (needed for sizes and overlays)")
    ap.add_argument("--labels", default=None, help="YOLO labels directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.30, help="IoU match threshold")
    ap.add_argument("--conf", type=float, default=0.25, help="Image-level conf threshold")
    ap.add_argument("--class-agnostic", action="store_true", default=True)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    preds_by_image = load_predictions(args.preds)

    samples = load_dataset(args.images, args.labels)
    gts_by_image = {s.image_path.name: s.boxes for s in samples if s.label_path is not None}

    if not gts_by_image:
        # No ground truth: qualitative only.
        msg = {
            "evaluable": False,
            "reason": "No ground-truth labels found. Quantitative detection metrics "
                      "require labelled data. Produced a qualitative prediction gallery instead.",
            "num_images_with_predictions": len(preds_by_image),
        }
        write_json(msg, out / "eval.json")
        print(msg["reason"])
        # Build a simple gallery from prediction overlays if present.
        gallery_dir = ensure_dir(out / "gallery")
        entries = []
        for s in samples:
            preds = preds_by_image.get(s.image_path.name, [])
            from netinspect.visualize import overlay_boxes
            img = read_image(s.image_path)
            rel = f"{s.stem}.jpg"
            write_image(gallery_dir / rel, overlay_boxes(img, preds=preds))
            entries.append({"image": rel, "caption": f"{s.image_path.name}: {len(preds)} pred(s)"})
        write_gallery_markdown(gallery_dir, entries, "Qualitative predictions (no ground truth)")
        return

    # Quantitative detection evaluation.
    det = evaluate_detection(preds_by_image, gts_by_image, args.iou, args.class_agnostic)
    sweep = confidence_sweep(preds_by_image, gts_by_image, args.iou,
                             class_agnostic=args.class_agnostic)
    img_level = evaluate_image_level(preds_by_image, gts_by_image, args.conf)
    best = best_f1_threshold(sweep)

    # Per-image FP/FN lists and match overlays.
    fp_dir = ensure_dir(out / "failures")
    sample_by_name = {s.image_path.name: s for s in samples}
    false_positives, false_negatives = [], []
    for m in det["matches"]:
        s = sample_by_name.get(m.image)
        if s is None:
            continue
        if m.fp:
            false_positives.append({"image": m.image, "count": m.fp})
        if m.fn:
            false_negatives.append({"image": m.image, "count": m.fn})
        if m.fp or m.fn:  # save failure overlay
            img = read_image(s.image_path)
            vis = overlay_match(img, preds_by_image.get(m.image, []), s.boxes,
                                m.fp_pred_idx, m.fn_gt_idx)
            write_image(fp_dir / f"{s.stem}_match.jpg", vis)

    report = {
        "evaluable": True,
        "detection": det["overall"],
        "image_level": img_level,
        "confidence_sweep": sweep,
        "best_f1_threshold": best,
        "per_image": det["per_image"],
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "note": "Class-agnostic IoU matching: measures localisation, not class naming.",
    }
    write_json(report, out / "eval.json")

    o = det["overall"]
    print("Detection (IoU >= {:.2f}, class-agnostic={})".format(args.iou, args.class_agnostic))
    print(f"  precision={o['precision']:.3f} recall={o['recall']:.3f} "
          f"f1={o['f1']:.3f} AP={o['ap']:.3f}")
    print(f"  TP={o['tp']} FP={o['fp']} FN={o['fn']} "
          f"(scored images={o['num_scored_images']})")
    print(f"Image-level: precision={img_level['precision']:.3f} "
          f"recall={img_level['recall']:.3f} accuracy={img_level['accuracy']:.3f}")
    if best:
        print(f"Best-F1 conf threshold: {best['conf_threshold']} (F1={best['f1']:.3f})")
    print(f"\nWrote {out / 'eval.json'}; failure overlays in {fp_dir}")


if __name__ == "__main__":
    main()

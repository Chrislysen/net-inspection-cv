"""Does test-time domain normalisation reduce different-day false alarms?

Underwater colour cast / lighting shift is the suspected driver of the seg model's
out-of-distribution false positives. A cheap thing to try is normalising it away at
inference: gray-world white balance + CLAHE (`preprocess.py`). This script measures
the different-day false-positive rate and damage recall with RAW vs NORMALISED
frames, for the detector and the segmenter.

Honest expectation: the models were trained on RAW frames, so normalising only at
test time is a train/test mismatch and may not help (or may hurt). The result is
reported either way — the point is to know, not to assume.

Example
-------
    python scripts/eval_domain_norm.py --out reports/results/domain_norm
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect.data import load_dataset
from netinspect.evaluate import evaluate_detection
from netinspect.model_baseline import YoloConfig, load_model, predict_image
from netinspect.preprocess import apply_clahe, gray_world_white_balance
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()
UNDAMAGED = "data/processed/solaqua_diffday"
COMPOSITED = ("data/processed/diffday_composite/images/test",
              "data/processed/diffday_composite/labels/test")


def _normalise(img):
    return apply_clahe(gray_world_white_balance(img))


def _fp_rate(model, imgs, conf, norm):
    n = sum(1 for p in imgs
            if any(b.score >= conf for b in predict_image(
                model, _normalise(read_image(p)) if norm else read_image(p),
                YoloConfig(conf=0.01, imgsz=480))))
    return round(n / max(1, len(imgs)), 3)


def _recall(model, samples, conf, iou, norm):
    preds, gts = {}, {}
    for s in samples:
        img = read_image(s.image_path)
        if norm:
            img = _normalise(img)
        preds[s.image_path.name] = [b for b in predict_image(model, img, YoloConfig(conf=0.01, imgsz=480))
                                    if b.score >= conf]
        gts[s.image_path.name] = s.boxes
    return round(evaluate_detection(preds, gts, iou)["overall"]["f1"], 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", default="models/yolo_damage_v1.pt")
    ap.add_argument("--seg", default="models/yolo_damage_seg_v3.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--out", default="reports/results/domain_norm")
    args = ap.parse_args()

    undamaged = list_images(UNDAMAGED)
    samples = load_dataset(*COMPOSITED)
    rows = {}
    for name, path in (("det v1", args.det), ("seg v3", args.seg)):
        model = load_model(path)
        LOGGER.info("Evaluating %s raw vs normalised...", name)
        rows[name] = {
            "fp_raw": _fp_rate(model, undamaged, args.conf, False),
            "fp_norm": _fp_rate(model, undamaged, args.conf, True),
            "recall_raw": _recall(model, samples, args.conf, args.iou, False),
            "recall_norm": _recall(model, samples, args.conf, args.iou, True),
        }

    out = ensure_dir(args.out)
    write_json({"conf": args.conf, "iou": args.iou, "frames": len(undamaged), "models": rows},
               out / "domain_norm.json")
    md = ["# Test-time domain normalisation (gray-world WB + CLAHE) — different day\n",
          f"{len(undamaged)} undamaged frames; conf={args.conf}, IoU={args.iou}.\n",
          "| Model | FP rate (raw) | FP rate (normalised) | Recall F1 (raw) | Recall F1 (normalised) |",
          "|---|---|---|---|---|"]
    for n, r in rows.items():
        md.append(f"| {n} | {r['fp_raw']:.0%} | {r['fp_norm']:.0%} | {r['recall_raw']} | {r['recall_norm']} |")
    md.append("\n> Models were trained on RAW frames, so this is test-time-only normalisation "
              "(a train/test mismatch). To actually exploit normalisation it must be applied in "
              "BOTH training and inference — a retrain. Reported as measured, not assumed.")
    (out / "domain_norm.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / 'domain_norm.md'}")


if __name__ == "__main__":
    main()

"""Evaluate the det-gated / seg-confirmed ensemble against each model alone.

Head-to-head on the same frames: the robust box detector (`det v1`), the best
segmenter (`seg v3`), and their **agreement ensemble** (det proposes, seg
confirms). Reports false-positive rate on REAL undamaged net (incl. the held-out
different day) and damage recall on composited test sets — the numbers that show
whether agreement keeps the detector's low OOD false alarms while adding masks.

Example
-------
    python scripts/eval_ensemble.py --det models/yolo_damage_v1.pt \\
        --seg models/yolo_damage_seg_v3.pt --out reports/results/ensemble
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect.data import load_dataset
from netinspect.ensemble import EnsembleConfig, combine
from netinspect.evaluate import evaluate_detection
from netinspect.model_baseline import YoloConfig, load_model, predict_image
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()

UNDAMAGED = {
    "bag1 (train backgrounds)": "data/processed/solaqua_frames",
    "bag2 (same site, other clip)": "data/processed/solaqua_bag2",
    "different DAY": "data/processed/solaqua_diffday",
}
COMPOSITED = {
    "in-clip": ("data/processed/real_composite/images/test", "data/processed/real_composite/labels/test"),
    "cross-clip (bag2)": ("data/processed/bag2_composite/images/test", "data/processed/bag2_composite/labels/test"),
    "different-day": ("data/processed/diffday_composite/images/test", "data/processed/diffday_composite/labels/test"),
}


def _variants(det_model, seg_model, img, cfg, imgsz):
    """Return {name: boxes} for det-only, seg-only, and the agreement ensemble."""
    det = predict_image(det_model, img, YoloConfig(conf=0.01, imgsz=imgsz))
    seg = predict_image(seg_model, img, YoloConfig(conf=0.01, imgsz=imgsz))
    return {
        "det v1": [b for b in det if b.score >= cfg.det_conf],
        "seg v3": [b for b in seg if b.score >= cfg.seg_conf],
        "ensemble (det∧seg)": combine(det, seg, cfg),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", default="models/yolo_damage_v1.pt")
    ap.add_argument("--seg", default="models/yolo_damage_seg_v3.pt")
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--agree-iou", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="reports/results/ensemble")
    args = ap.parse_args()

    cfg = EnsembleConfig(det_conf=args.det_conf, seg_conf=args.seg_conf,
                         agree_iou=args.agree_iou, mode="agree")
    det_model = load_model(args.det)
    seg_model = load_model(args.seg)
    names = ["det v1", "seg v3", "ensemble (det∧seg)"]

    LOGGER.info("False positives on undamaged net...")
    fp = {n: {} for n in names}
    for set_name, d in UNDAMAGED.items():
        imgs = list_images(d)
        if not imgs:
            continue
        counts = {n: [] for n in names}
        for p in imgs:
            v = _variants(det_model, seg_model, read_image(p), cfg, args.imgsz)
            for n in names:
                counts[n].append(len(v[n]))
        for n in names:
            c = counts[n]
            fp[n][set_name] = {"frames": len(c), "detections": sum(c),
                               "mean_per_frame": round(sum(c) / len(c), 3),
                               "fp_frame_rate": round(sum(1 for x in c if x > 0) / len(c), 3)}

    LOGGER.info("Damage recall on composited sets...")
    rec = {n: {} for n in names}
    for set_name, (imgs, lbls) in COMPOSITED.items():
        if not list_images(imgs):
            continue
        samples = load_dataset(imgs, lbls)
        preds = {n: {} for n in names}
        gts = {}
        for s in samples:
            v = _variants(det_model, seg_model, read_image(s.image_path), cfg, args.imgsz)
            for n in names:
                preds[n][s.image_path.name] = v[n]
            gts[s.image_path.name] = s.boxes
        for n in names:
            r = evaluate_detection(preds[n], gts, args.iou)["overall"]
            rec[n][set_name] = {k: round(r[k], 3) for k in ("precision", "recall", "f1")}

    out = ensure_dir(args.out)
    write_json({"config": vars(args), "false_positives_on_undamaged": fp,
                "recall_by_distance": rec}, out / "ensemble.json")

    md = ["# Ensemble — robust detector proposes, segmenter confirms\n",
          f"det={args.det.split('/')[-1]}, seg={args.seg.split('/')[-1]}, "
          f"agree_iou={args.agree_iou}, conf={args.det_conf}\n",
          "## False positives on REAL UNDAMAGED net (lower is better)\n",
          "| Model | " + " | ".join(UNDAMAGED.keys()) + " |",
          "|---|" + "---|" * len(UNDAMAGED)]
    for n in names:
        cells = [f"{fp[n][s]['fp_frame_rate']:.0%}" if s in fp[n] else "—" for s in UNDAMAGED]
        md.append(f"| {n} | " + " | ".join(cells) + " |")
    md += ["\n## Damage recall (F1) on composited test sets\n",
           "| Model | " + " | ".join(COMPOSITED.keys()) + " |",
           "|---|" + "---|" * len(COMPOSITED)]
    for n in names:
        cells = [f"{rec[n][s]['f1']}" if s in rec[n] else "—" for s in COMPOSITED]
        md.append(f"| {n} | " + " | ".join(cells) + " |")
    md.append("\n> Agreement of two independently-trained models suppresses model-specific "
              "spurious cues. Both models are trained on synthetic damage on real backgrounds: "
              "this strengthens proxy robustness, not validated real-damage accuracy.")
    (out / "ensemble.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / 'ensemble.md'}")


if __name__ == "__main__":
    main()

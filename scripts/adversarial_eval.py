"""Adversarial "is it cheating?" evaluation for the trained detector.

A high F1 on synthetic-damage-on-real-backgrounds is suspicious: the model could
be keying on compositing artifacts or background cues rather than damage. This
script runs the tests that try to *break* that interpretation:

1. **False positives on REAL UNDAMAGED net.** Real net has no dark see-through
   holes, so a damage detector should *rarely* fire. If it fires a lot on
   undamaged net (especially the very backgrounds it trained on), it is keying on
   something spurious. Run on bag1 (train backgrounds), bag2 (same site, other
   clip), and a different-DAY clip.
2. **Generalisation to a different DAY.** Recall on damage composited onto
   different-day frames (backgrounds the model never saw).
3. **FROC operating curve.** False-positives-per-undamaged-frame vs recall-on-
   damage across confidence thresholds — the curve an operator actually picks on.

Honesty: even passing all of these does NOT prove real-damage performance (the
damage is still synthetic). It only rules out the cheapest ways of cheating and
characterises behaviour. Real labelled damage remains required.

Example
-------
    python scripts/adversarial_eval.py --yolo-weights models/yolo_damage_v1.pt --out reports/results/adversarial
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect.classical_baseline import ClassicalConfig
from netinspect.data import load_dataset
from netinspect.evaluate import evaluate_detection
from netinspect.inference import NetInspector
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()

# Real UNDAMAGED frame sets (no damage present -> any detection is a false alarm).
UNDAMAGED = {
    "bag1 (train backgrounds)": "data/processed/solaqua_frames",
    "bag2 (same site, other clip)": "data/processed/solaqua_bag2",
    "different DAY": "data/processed/solaqua_diffday",
}
# Composited (labelled) test sets at increasing background distance from training.
COMPOSITED = {
    "in-clip": ("data/processed/real_composite/images/test", "data/processed/real_composite/labels/test"),
    "cross-clip (bag2)": ("data/processed/bag2_composite/images/test", "data/processed/bag2_composite/labels/test"),
    "different-day": ("data/processed/diffday_composite/images/test", "data/processed/diffday_composite/labels/test"),
}


def _fp_on_undamaged(insp, method, conf):
    rows = {}
    for name, d in UNDAMAGED.items():
        imgs = list_images(d)
        if not imgs:
            continue
        dets = [len(insp.predict(read_image(p), method=method, conf=conf).boxes) for p in imgs]
        rows[name] = {"frames": len(imgs), "total_detections": sum(dets),
                      "mean_per_frame": round(sum(dets) / len(imgs), 3),
                      "frames_with_fp": sum(1 for x in dets if x > 0),
                      "fp_frame_rate": round(sum(1 for x in dets if x > 0) / len(imgs), 3)}
    return rows


def _recall_by_distance(insp, method, conf, iou):
    rows = {}
    for name, (imgs, lbls) in COMPOSITED.items():
        if not list_images(imgs):
            continue
        samples = load_dataset(imgs, lbls)
        preds = {s.image_path.name: insp.predict(read_image(s.image_path), method=method, conf=conf).boxes
                 for s in samples}
        gts = {s.image_path.name: s.boxes for s in samples}
        r = evaluate_detection(preds, gts, iou)["overall"]
        rows[name] = {k: round(r[k], 3) for k in ("precision", "recall", "f1", "ap")}
    return rows


def _froc(insp, method, iou):
    """FP-per-undamaged-frame vs recall-on-damage across confidence thresholds."""
    und = list_images(UNDAMAGED["different DAY"]) or list_images(UNDAMAGED["bag2 (same site, other clip)"])
    samples = load_dataset(*COMPOSITED["different-day"])
    und_imgs = [read_image(p) for p in und]
    dmg = [(read_image(s.image_path), s.boxes) for s in samples]
    curve = []
    for conf in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fp = sum(len(insp.predict(im, method=method, conf=conf).boxes) for im in und_imgs)
        preds = {i: insp.predict(im, method=method, conf=conf).boxes for i, (im, _) in enumerate(dmg)}
        gts = {i: bx for i, (_, bx) in enumerate(dmg)}
        r = evaluate_detection(preds, gts, iou)["overall"]
        curve.append({"conf": conf, "fp_per_undamaged_frame": round(fp / max(1, len(und_imgs)), 3),
                      "recall": round(r["recall"], 3), "precision": round(r["precision"], 3)})
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--method", default="yolo", choices=["yolo", "classical", "patchcore"])
    ap.add_argument("--patchcore-model", default=None)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--out", default="reports/results/adversarial")
    args = ap.parse_args()

    ccfg = ClassicalConfig()
    if Path(args.config).exists():
        params = _common.load_yaml(args.config).get("classical", {})
        ccfg = ClassicalConfig(**{k: v for k, v in params.items() if k in ClassicalConfig().__dict__})
    insp = NetInspector(classical_cfg=ccfg, patchcore_model_path=args.patchcore_model,
                        yolo_weights=args.yolo_weights)
    if args.method not in insp.available_methods():
        print(f"Method '{args.method}' unavailable ({insp.available_methods()}).")
        return

    LOGGER.info("FP on real undamaged net...")
    fp = _fp_on_undamaged(insp, args.method, args.conf)
    LOGGER.info("Recall by background distance...")
    rec = _recall_by_distance(insp, args.method, args.conf, args.iou)
    LOGGER.info("FROC curve...")
    froc = _froc(insp, args.method, args.iou)

    out = ensure_dir(args.out)
    report = {"method": args.method, "conf": args.conf, "iou": args.iou,
              "false_positives_on_undamaged": fp, "recall_by_distance": rec, "froc": froc}
    write_json(report, out / "adversarial.json")

    md = [f"# Adversarial evaluation — `{args.method}`\n",
          "## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)\n",
          "| Frame set | Frames | Detections | Mean/frame | FP frame rate |",
          "|---|---|---|---|---|"]
    for name, r in fp.items():
        md.append(f"| {name} | {r['frames']} | {r['total_detections']} | {r['mean_per_frame']} | {r['fp_frame_rate']:.0%} |")
    md += ["\n## 2. Damage recall by background distance (composited damage)\n",
           "| Background | Precision | Recall | F1 | AP |", "|---|---|---|---|---|"]
    for name, r in rec.items():
        md.append(f"| {name} | {r['precision']} | {r['recall']} | {r['f1']} | {r['ap']} |")
    md += ["\n## 3. FROC (different-day): FP per undamaged frame vs recall\n",
           "| conf | FP/undamaged frame | recall | precision |", "|---|---|---|---|"]
    for c in froc:
        md.append(f"| {c['conf']} | {c['fp_per_undamaged_frame']} | {c['recall']} | {c['precision']} |")
    md.append("\n> Passing these rules out the cheapest cheating (background/artifact keying) and "
              "characterises behaviour. It does NOT prove real-damage performance — the damage is "
              "still synthetic. Real labelled damage remains required.")
    (out / "adversarial.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    print(f"\nWrote {out / 'adversarial.md'}")


if __name__ == "__main__":
    main()

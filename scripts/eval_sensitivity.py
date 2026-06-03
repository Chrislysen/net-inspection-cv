"""Sensitivity sweep: damage recall vs false-alarm rate across confidence thresholds.

"Make it more sensitive" = lower the confidence threshold so the detector flags
fainter/smaller damage — at the cost of more false alarms on undamaged net. This
script makes that trade-off explicit: for one model it reports, at each threshold,
damage **recall** on a composited test set and the **false-positive frame rate** on
real undamaged net. Run it on the normal *and* the `--hard` subtle-damage set to
see how much sensitivity you must trade to keep catching subtle damage.

Honesty: damage is synthetic, so this characterises behaviour and picks an
*operating point*; it does not validate real-damage accuracy. For net inspection a
missed hole (fish escape) usually costs more than a false alarm, which argues for a
sensitive point + human review — but that is a stakeholder decision, not a default.

Example
-------
    python scripts/eval_sensitivity.py --yolo-weights models/yolo_damage_seg_gpu.pt \\
        --composited data/processed/hard_composite/images/test data/processed/hard_composite/labels/test \\
        --undamaged data/processed/solaqua_diffday --out reports/results/sensitivity_hard
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect.data import load_dataset
from netinspect.evaluate import evaluate_detection
from netinspect.model_baseline import YoloConfig, load_model, predict_image
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()
CONFS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--composited", nargs=2, metavar=("IMAGES", "LABELS"), required=True)
    ap.add_argument("--undamaged", required=True, help="Dir of real undamaged frames")
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--out", default="reports/results/sensitivity")
    args = ap.parse_args()

    model = load_model(args.yolo_weights)
    samples = load_dataset(*args.composited)
    undamaged = list_images(args.undamaged)
    LOGGER.info("Sensitivity sweep: %d composited, %d undamaged frames", len(samples), len(undamaged))

    # Predict once at conf 0.01, then threshold in-memory per conf level.
    dmg = [(s.image_path.name, s.boxes,
            predict_image(model, read_image(s.image_path), YoloConfig(conf=0.01, imgsz=args.imgsz)))
           for s in samples]
    und = [predict_image(model, read_image(p), YoloConfig(conf=0.01, imgsz=args.imgsz))
           for p in undamaged]

    rows = []
    for c in CONFS:
        preds = {n: [b for b in bs if b.score >= c] for n, _, bs in dmg}
        gts = {n: g for n, g, _ in dmg}
        r = evaluate_detection(preds, gts, args.iou)["overall"]
        fp_frames = sum(1 for bs in und if any(b.score >= c for b in bs))
        rows.append({"conf": c, "recall": round(r["recall"], 3),
                     "precision": round(r["precision"], 3), "f1": round(r["f1"], 3),
                     "undamaged_fp_rate": round(fp_frames / max(1, len(und)), 3)})

    out = ensure_dir(args.out)
    write_json({"weights": args.yolo_weights, "composited": args.composited[0],
                "undamaged": args.undamaged, "sweep": rows}, out / "sensitivity.json")
    md = [f"# Sensitivity sweep — `{args.yolo_weights.split('/')[-1]}`\n",
          f"Damage: `{args.composited[0]}` ({len(samples)} frames) · undamaged: "
          f"`{args.undamaged}` ({len(undamaged)} frames). Lower conf = more sensitive.\n",
          "| conf | damage recall | precision | F1 | undamaged FP rate |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['conf']} | {r['recall']} | {r['precision']} | {r['f1']} | "
                  f"{r['undamaged_fp_rate']:.0%} |")
    md.append("\n> Sensitivity is a dial, not a fact: lower threshold catches more (faint) damage "
              "but raises false alarms. Damage is synthetic — pick the operating point with "
              "stakeholders on REAL labelled data.")
    (out / "sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / 'sensitivity.md'}")


if __name__ == "__main__":
    main()

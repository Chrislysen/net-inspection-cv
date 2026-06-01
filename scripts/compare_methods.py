"""Compare classical / anomaly / YOLO on one labelled test set, side by side.

Runs whichever methods are available on the same images, evaluates each with the
shared IoU-matching metrics, and writes a comparison table (JSON + markdown).
This is the honest "which approach is most promising" deliverable.

Example
-------
    python scripts/compare_methods.py \\
        --images data/processed/real_composite/images/test \\
        --labels data/processed/real_composite/labels/test \\
        --anomaly-model outputs/anomaly/model \\
        --yolo-weights runs/detect/train/weights/best.pt \\
        --out outputs/comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect.classical_baseline import ClassicalConfig
from netinspect.classical_baseline import detect as classical_detect
from netinspect.data import load_dataset
from netinspect.evaluate import coco_map, evaluate_detection, evaluate_image_level
from netinspect.utils import ensure_dir, get_logger, read_image, write_json

LOGGER = get_logger()


def _classical(samples, cfg):
    return {s.image_path.name: classical_detect(read_image(s.image_path), cfg).boxes
            for s in samples}


def _anomaly(samples, model_path):
    from netinspect.anomaly import AnomalyModel, score_image
    model = AnomalyModel.load(model_path)
    return {s.image_path.name: score_image(read_image(s.image_path), model).boxes
            for s in samples}


def _patchcore(samples, model_path):
    from netinspect.patchcore import PatchCoreModel, score_image
    model = PatchCoreModel.load(model_path)
    return {s.image_path.name: score_image(read_image(s.image_path), model).boxes
            for s in samples}


def _yolo(samples, weights):
    from netinspect.model_baseline import YoloConfig, load_model, predict_image
    cfg = YoloConfig(conf=0.25, iou=0.5)
    model = load_model(weights)
    return {s.image_path.name: predict_image(model, read_image(s.image_path), cfg)
            for s in samples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="classical config yaml")
    ap.add_argument("--anomaly-model", default=None)
    ap.add_argument("--patchcore-model", default=None)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--iou", type=float, default=0.30)
    args = ap.parse_args()

    samples = load_dataset(args.images, args.labels)
    labelled = [s for s in samples if s.label_path is not None]
    if not labelled:
        print("No labels found; cannot compare quantitatively.")
        return
    gts = {s.image_path.name: s.boxes for s in samples}

    ccfg = ClassicalConfig()
    if args.config:
        params = _common.load_yaml(args.config).get("classical", {})
        ccfg = ClassicalConfig(**{k: v for k, v in params.items() if k in ClassicalConfig().__dict__})

    methods: dict[str, dict] = {}
    runners = [("classical", lambda: _classical(samples, ccfg))]
    if args.anomaly_model:
        runners.append(("anomaly", lambda: _anomaly(samples, args.anomaly_model)))
    if args.patchcore_model:
        runners.append(("patchcore", lambda: _patchcore(samples, args.patchcore_model)))
    if args.yolo_weights:
        runners.append(("yolo", lambda: _yolo(samples, args.yolo_weights)))

    rows = []
    for name, run in runners:
        try:
            preds = run()
        except Exception as exc:  # missing dep / weights
            LOGGER.warning("Skipping %s: %s", name, exc)
            continue
        det = evaluate_detection(preds, gts, args.iou)["overall"]
        img = evaluate_image_level(preds, gts, conf_threshold=0.25)
        cmap = coco_map(preds, gts)
        methods[name] = {"detection": det, "image_level": img, "coco_map": cmap}
        rows.append((name, det, img, cmap))

    out = ensure_dir(args.out)
    write_json({"iou": args.iou, "num_images": len(samples), "methods": methods},
               out / "comparison.json")

    # Markdown table.
    md = ["# Method comparison",
          f"\nTest set: `{args.images}` ({len(samples)} images, IoU={args.iou}, class-agnostic)\n",
          "| Method | Precision | Recall | F1 | AP@.5 | mAP@[.5:.95] | Image-level acc |",
          "|---|---|---|---|---|---|---|"]
    for name, det, img, cmap in rows:
        md.append(f"| {name} | {det['precision']:.3f} | {det['recall']:.3f} | "
                  f"{det['f1']:.3f} | {det['ap']:.3f} | {cmap['map_50_95']:.3f} | "
                  f"{img['accuracy']:.3f} |")
    md.append("\n> Numbers are on **synthetic damage composited on real backgrounds** "
              "from a single clip. Optimistic vs. truly independent sites; treat as "
              "relative comparison, not validated absolute performance.")
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md[2:]))
    print(f"\nWrote {out / 'comparison.md'}")


if __name__ == "__main__":
    main()

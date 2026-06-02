"""SSL vs supervised features for one-class anomaly detection (DINOv2 vs ResNet).

The PatchCore anomaly detector (``patchcore.py``) needs only *normal* frames —
exactly SOLAQUA's situation. Its only learned component is the **backbone** that
turns patches into feature vectors. This script asks one focused question:

    Do off-the-shelf **self-supervised** features (DINOv2, trained without labels)
    transfer better to underwater net imagery than **ImageNet-supervised**
    features (ResNet18), for the *same* anomaly detector?

It fits nothing — pass two pre-fit PatchCore models (see train_patchcore.py with
``--backbone resnet18`` and ``--backbone dinov2_vits14``) and it evaluates both
identically on:
  * **different-day composited damage** — recall/precision/F1 on the hardest,
    most out-of-distribution backgrounds (the real generalisation test);
  * **in-clip composited damage** — a same-distribution reference;
  * **different-day UNDAMAGED net** — false-alarm rate (every detection is wrong).

Honesty: DINOv2 here is pretrained on natural images, NOT on SOLAQUA. This tests
*transfer of published SSL features*, not the deferred experiment of pretraining
DINO/MAE on the unlabelled SOLAQUA frames (GPU-bound; still the next step). Every
score is on SYNTHETIC damage and on *undamaged* net — it characterises behaviour,
not validated real-damage performance.

Example
-------
    python scripts/compare_anomaly_backbones.py \\
        --models resnet18=models/patchcore_resnet18 dinov2=models/patchcore_dino_vits14 \\
        --out reports/results/ssl_dino
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect.data import load_dataset
from netinspect.evaluate import evaluate_detection
from netinspect.inference import NetInspector
from netinspect.utils import ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()

# Composited (labelled) damage test sets, easy -> hard background distance, each
# paired with the matching REAL UNDAMAGED frames from the same clip/day. The
# undamaged set is both the AUROC negatives and the false-alarm probe.
SETS = {
    "in-clip": {
        "imgs": "data/processed/real_composite/images/test",
        "lbls": "data/processed/real_composite/labels/test",
        "undamaged": "data/processed/solaqua_frames",
    },
    "different-day": {
        "imgs": "data/processed/diffday_composite/images/test",
        "lbls": "data/processed/diffday_composite/labels/test",
        "undamaged": "data/processed/solaqua_diffday",
    },
}


def _score_set(insp, conf):
    """Score every frame once; return per-set boxes, GT, and max anomaly scores."""
    data = {}
    for name, s in SETS.items():
        if not list_images(s["imgs"]):
            continue
        samples = load_dataset(s["imgs"], s["lbls"])
        preds, gts, pos_scores = {}, {}, []
        for sm in samples:
            r = insp.predict(read_image(sm.image_path), method="patchcore", conf=conf)
            preds[sm.image_path.name] = r.boxes
            gts[sm.image_path.name] = sm.boxes
            if sm.boxes:                      # frame actually contains damage
                pos_scores.append(r.meta["max_score"])
        und = list_images(s["undamaged"])
        und_res = [insp.predict(read_image(p), method="patchcore", conf=conf) for p in und]
        neg_scores = [r.meta["max_score"] for r in und_res]
        det_counts = [len(r.boxes) for r in und_res]
        data[name] = {"preds": preds, "gts": gts, "pos_scores": pos_scores,
                      "neg_scores": neg_scores, "det_counts": det_counts,
                      "n_undamaged": len(und)}
    return data


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    """Threshold-free separability of damaged vs undamaged frames (Mann-Whitney)."""
    if not pos or not neg:
        return None
    from sklearn.metrics import roc_auc_score
    y = [1] * len(pos) + [0] * len(neg)
    return round(float(roc_auc_score(y, pos + neg)), 3)


def _summarise(data, iou):
    rows = {}
    for name, d in data.items():
        r = evaluate_detection(d["preds"], d["gts"], iou)["overall"]
        det = d["det_counts"]
        rows[name] = {
            "image_auroc": _auroc(d["pos_scores"], d["neg_scores"]),
            "localisation": {k: round(r[k], 3) for k in ("precision", "recall", "f1", "ap")},
            "fp_on_undamaged": {
                "frames": d["n_undamaged"], "total_detections": sum(det),
                "mean_per_frame": round(sum(det) / max(1, len(det)), 3),
                "fp_frame_rate": round(sum(1 for x in det if x > 0) / max(1, len(det)), 3)},
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, nargs="+",
                    help="label=model_path entries (path without .npz)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--out", default="reports/results/ssl_dino")
    args = ap.parse_args()

    results = {}
    for entry in args.models:
        label, _, path = entry.partition("=")
        if not path:
            ap.error(f"--models entries must be label=path, got '{entry}'")
        LOGGER.info("Evaluating backbone '%s' (%s)...", label, path)
        insp = NetInspector(patchcore_model_path=path)
        if "patchcore" not in insp.available_methods():
            print(f"patchcore unavailable for {path} (need torch + the .npz). Skipping.")
            continue
        results[label] = {"model": path,
                          "by_set": _summarise(_score_set(insp, args.conf), args.iou)}

    out = ensure_dir(args.out)
    write_json({"conf": args.conf, "iou": args.iou, "backbones": results},
               out / "ssl_dino.json")

    md = ["# Self-supervised vs supervised features for anomaly detection\n",
          "Same PatchCore detector, same normal training frames — only the patch "
          "**backbone** differs. DINOv2 features are self-supervised (no labels); "
          "ResNet18 features are ImageNet-supervised.\n",
          "**Headline metric is image-level AUROC** — threshold-free separability of "
          "damaged vs undamaged frames by the anomaly score. It is the fair test of "
          "feature quality because it does not depend on a threshold (and the threshold "
          "is exactly what fails to transfer across days). Localisation/FP columns use "
          f"the default 2x-median threshold at conf={args.conf}, IoU={args.iou}.\n",
          "## Image-level AUROC (damaged vs undamaged frames) — higher is better\n",
          "| Backbone | in-clip | different-day |", "|---|---|---|"]
    for label, r in results.items():
        a = r["by_set"]
        md.append(f"| {label} | {a.get('in-clip', {}).get('image_auroc')} | "
                  f"{a.get('different-day', {}).get('image_auroc')} |")
    md += ["\n## Localisation of (synthetic) damage — boxes at the default threshold\n",
           "| Backbone | Background | Precision | Recall | F1 | AP |",
           "|---|---|---|---|---|---|"]
    for label, r in results.items():
        for bg, m in r["by_set"].items():
            loc = m["localisation"]
            md.append(f"| {label} | {bg} | {loc['precision']} | {loc['recall']} | "
                      f"{loc['f1']} | {loc['ap']} |")
    md += ["\n## False alarms on REAL UNDAMAGED net (default threshold)\n",
           "| Backbone | Set | Frames | Mean det/frame | FP frame rate |",
           "|---|---|---|---|---|"]
    for label, r in results.items():
        for bg, m in r["by_set"].items():
            fp = m["fp_on_undamaged"]
            md.append(f"| {label} | {bg} | {fp['frames']} | {fp['mean_per_frame']} | "
                      f"{fp['fp_frame_rate']:.0%} |")
    md.append("\n> DINOv2 here is pretrained on natural images, NOT on SOLAQUA — this "
              "measures transfer of published self-supervised features, not the deferred "
              "SOLAQUA-pretraining experiment. Damage is synthetic and the net is "
              "undamaged: this characterises behaviour, not real-damage performance.")
    (out / "ssl_dino.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / 'ssl_dino.md'}")


if __name__ == "__main__":
    main()

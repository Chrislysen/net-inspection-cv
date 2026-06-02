"""Render result plots from the committed JSON result files.

Reads `reports/results/**` and writes PNG figures to `reports/results/plots/`,
which are embedded in the README/report. Regenerate after re-running the
evaluation scripts. Uses a non-interactive backend (no display needed).

Example
-------
    python scripts/make_plots.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import _common  # noqa: F401
import matplotlib.pyplot as plt  # noqa: E402

from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()
RESULTS = _common.REPO_ROOT / "reports" / "results"
C = {"classical": "#4C72B0", "anomaly": "#937860", "patchcore": "#55A868",
     "yolo": "#C44E52", "v1": "#C44E52", "v2": "#DD8452", "v3": "#55A868",
     "v4": "#8172B3"}


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def plot_label_free_ladder(out: Path):
    d = _load(RESULTS / "comparison_4way" / "comparison.json")
    if not d:
        return
    order = ["anomaly", "classical", "patchcore", "yolo"]
    labels = {"anomaly": "Hand-crafted\nanomaly", "classical": "Classical\nheuristic",
              "patchcore": "PatchCore\n(foundation)", "yolo": "YOLOv8\n(supervised)"}
    needs = {"anomaly": "label-free", "classical": "label-free",
             "patchcore": "label-free", "yolo": "supervised"}
    f1 = [d["methods"][m]["detection"]["f1"] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([labels[m] for m in order], f1, color=[C[m] for m in order])
    for b, m in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center", fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.03, needs[m], ha="center",
                color="white", fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 (composite test, IoU 0.30)")
    ax.set_title("Damage-localisation F1 by method\n(label-free methods improve toward the supervised ceiling)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "method_f1_ladder.png", dpi=130); plt.close(fig)


def plot_method_metrics(out: Path):
    d = _load(RESULTS / "comparison_4way" / "comparison.json")
    if not d:
        return
    order = [m for m in ("classical", "anomaly", "patchcore", "yolo") if m in d["methods"]]
    metrics = ["precision", "recall", "f1", "ap"]
    import numpy as np
    x = np.arange(len(metrics)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(order):
        vals = [d["methods"][m]["detection"][k] for k in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=C.get(m, None))
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend(ncol=4, fontsize=9)
    ax.set_title("Detection metrics by method (in-clip composite test)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "method_metrics.png", dpi=130); plt.close(fig)


def plot_froc(out: Path):
    v1 = _load(RESULTS / "adversarial_yolo" / "adversarial.json")
    v2 = _load(RESULTS / "adversarial_seg" / "adversarial.json")
    v3 = _load(RESULTS / "adversarial_seg_v3" / "adversarial.json")
    v4 = _load(RESULTS / "adversarial_seg_v4" / "adversarial.json")
    if not v1:
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for d, name, col in ((v1, "det v1", C["v1"]), (v2, "seg v2 (1 clip)", C["v2"]),
                         (v3, "seg v3 (multi-clip)", C["v3"]),
                         (v4, "seg v4 (+strong aug)", C["v4"])):
        if not d:
            continue
        froc = d["froc"]
        ax.plot([p["fp_per_undamaged_frame"] for p in froc],
                [p["recall"] for p in froc], "-o", color=col, label=name)
        for p in froc:
            if p["conf"] in (0.3, 0.5, 0.7):
                ax.annotate(f"{p['conf']}", (p["fp_per_undamaged_frame"], p["recall"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("False positives per undamaged frame (different-day)")
    ax.set_ylabel("Recall on damage")
    ax.set_title("FROC — operating curve (labels = conf threshold)\nleft/up is better; v3≈v4 recover most of seg v2's OOD gap; det v1 still best")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "froc.png", dpi=130); plt.close(fig)


def plot_fp_undamaged(out: Path):
    v1 = _load(RESULTS / "adversarial_yolo" / "adversarial.json")
    v2 = _load(RESULTS / "adversarial_seg" / "adversarial.json")
    v3 = _load(RESULTS / "adversarial_seg_v3" / "adversarial.json")
    v4 = _load(RESULTS / "adversarial_seg_v4" / "adversarial.json")
    if not v1:
        return
    import numpy as np
    sets = list(v1["false_positives_on_undamaged"].keys())
    x = np.arange(len(sets)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    series = [("det v1", v1, C["v1"], -1.5), ("seg v2 (1 clip)", v2, C["v2"], -0.5),
              ("seg v3 (multi-clip)", v3, C["v3"], 0.5),
              ("seg v4 (+strong aug)", v4, C["v4"], 1.5)]
    for label, d, col, off in series:
        if not d:
            continue
        rates = [d["false_positives_on_undamaged"][s]["fp_frame_rate"] for s in sets]
        ax.bar(x + off * w, rates, w, label=label, color=col)
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" (", "\n(") for s in sets], fontsize=8)
    ax.set_ylabel("False-positive frame rate")
    ax.set_title("False positives on REAL UNDAMAGED net (lower is better)\nseg v2 regressed OOD; v3 cut it 31%->18%; stronger aug (v4) did not improve it")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fp_on_undamaged.png", dpi=130); plt.close(fig)


def plot_ensemble(out: Path):
    d = _load(RESULTS / "ensemble" / "ensemble.json")
    if not d:
        return
    import numpy as np
    models = ["det v1", "seg v3", "ensemble (det∧seg)"]
    fp = d["false_positives_on_undamaged"]
    rec = d["recall_by_distance"]
    # Different-day is the OOD test that matters; show FP rate and recall side by side.
    fp_rates = [fp[m].get("different DAY", {}).get("fp_frame_rate", 0) for m in models]
    recalls = [rec[m].get("different-day", {}).get("f1", 0) for m in models]
    cols = ["#C44E52", "#55A868", "#8172B3"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    b1 = a1.bar(range(len(models)), fp_rates, color=cols)
    a1.set_title("Different-day FALSE-POSITIVE rate\n(undamaged net — lower is better)")
    a1.set_ylabel("FP frame rate")
    for b, v in zip(b1, fp_rates):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.0%}", ha="center", fontweight="bold")
    b2 = a2.bar(range(len(models)), recalls, color=cols)
    a2.set_title("Different-day damage RECALL (F1)\n(higher is better)")
    a2.set_ylabel("F1")
    for b, v in zip(b2, recalls):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    for ax in (a1, a2):
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    a1.set_ylim(0, max(0.25, max(fp_rates) * 1.25)); a2.set_ylim(0, 1.05)
    fig.suptitle("Ensemble = detector's robustness + segmenter's masks (no retraining)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "ensemble_comparison.png", dpi=130); plt.close(fig)


def plot_temporal(out: Path):
    d = _load(RESULTS / "temporal" / "temporal.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(["raw\n(per-frame)", "temporally\nconfirmed"],
                  [d["raw_detections"], d["confirmed_detections"]],
                  color=["#BBBBBB", C["classical"]])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                int(b.get_height()), ha="center", fontweight="bold")
    ax.set_ylabel("False alarms over 120 undamaged frames")
    ax.set_title(f"Temporal confirmation removes "
                 f"{d['detection_reduction']:.0%} of transient false alarms")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "temporal_reduction.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS / "plots"))
    args = ap.parse_args()
    out = ensure_dir(args.out)
    made = []
    for fn in (plot_label_free_ladder, plot_method_metrics, plot_froc,
               plot_fp_undamaged, plot_ensemble, plot_temporal):
        try:
            fn(out)
            made.append(fn.__name__)
        except Exception as exc:  # keep going if one result file is absent
            LOGGER.warning("%s failed: %s", fn.__name__, exc)
    pngs = sorted(p.name for p in out.glob("*.png"))
    print(f"Wrote {len(pngs)} plots to {out}:")
    for p in pngs:
        print(f"  {p}")


if __name__ == "__main__":
    main()

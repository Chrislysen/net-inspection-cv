"""Build the README evidence figure: what actually drives false alarms.

One figure carrying the project's central finding, designed to be understood
without reading the surrounding text.

Design notes, since they were deliberate:

* **Colour means "model" everywhere it appears.** Panels A and C both encode the
  three detectors in the same three hues. Panel B needs clip identity instead, so
  it uses *position* (clips on the x-axis) rather than recolouring — reusing the
  same hues for a different variable in the middle panel would silently break the
  reader's colour-to-model mapping.
* **Zero is drawn, not omitted.** Three of the twelve bars are exactly 0%, and a
  bar of height zero reads as missing data. Each is given a visible baseline tick
  and an explicit "0%" label, because "this clip produced no false alarms" is a
  result, not an absence of one.
* **Palette is the validated categorical default** (blue / orange / aqua), which
  passes CVD separation. The aqua's contrast against the surface is below 3:1, so
  every bar carries a direct value label — the required relief.

Run: ``python scripts/make_evidence_figure.py``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()

FRAME_CONDITIONS = "reports/results/operating_envelope/frame_conditions.parquet"

# Validated categorical palette (light surface). Colour == model, everywhere.
MODEL_COLOURS = {"det_v1": "#2a78d6", "seg_v3": "#eb6834", "seg_gpu": "#1baf7a"}
MODEL_LABELS = {"det_v1": "detector (det v1)",
                "seg_v3": "segmenter (seg v3)",
                "seg_gpu": "segmenter (seg-gpu)"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#8a8981"
NEUTRAL = "#b9b8ae"


def _style(ax):
    """Recessive axes: the data carries the ink, not the frame."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_FAINT)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis="y", color=NEUTRAL, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)


def build(df, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODEL_COLOURS if f"fp_{m}" in df.columns]
    clips = sorted(df["clip"].unique())
    short = {c: c[-8:] for c in clips}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.28, top=0.74, bottom=0.20, left=0.055, right=0.985)

    # ---- Panel A: false alarms are a property of the clip -------------------
    ax = axes[0]
    _style(ax)
    x = np.arange(len(clips))
    width = 0.26
    for i, m in enumerate(models):
        rates = [df.loc[df["clip"] == c, f"fp_{m}"].mean() for c in clips]
        pos = x + (i - (len(models) - 1) / 2) * width
        ax.bar(pos, rates, width * 0.88, color=MODEL_COLOURS[m],
               label=MODEL_LABELS[m], zorder=3)
        for px, r in zip(pos, rates):
            if r < 0.005:
                # Zero is a finding — draw it and say so.
                ax.plot([px - width * 0.44, px + width * 0.44], [0, 0],
                        color=MODEL_COLOURS[m], lw=2.4, zorder=4,
                        solid_capstyle="butt")
                ax.text(px, 0.012, "0%", ha="center", va="bottom", fontsize=7.5,
                        color=INK_SOFT, zorder=5)
            else:
                ax.text(px, r + 0.012, f"{r:.0%}", ha="center", va="bottom",
                        fontsize=8, color=INK, zorder=5,
                        fontweight="bold" if r > 0.25 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{short[c]}\n{c[:10]}\n{df.loc[df['clip'] == c, 'standoff'].mean():.2f} m"
         for c in clips], fontsize=8.5, color=INK_SOFT)
    ax.set_ylabel("frames with ≥1 false alarm", fontsize=9.5, color=INK_SOFT)
    ax.set_ylim(0, 0.42)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("A · False alarms belong to the CLIP, not the day",
                 fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")

    same_day = [c for c in clips if c.startswith("2024-08-22")]
    if len(same_day) >= 2:
        lo = x[clips.index(same_day[0])] - 0.42
        hi = x[clips.index(same_day[-1])] + 0.42
        ax.annotate("", xy=(lo, 0.392), xytext=(hi, 0.392),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4",
                                    color=INK_FAINT, lw=0.9))
        ax.text((lo + hi) / 2, 0.400,
                "same day · same site · same camera · same standoff",
                ha="center", va="bottom", fontsize=8, color=INK_SOFT, style="italic")

    # ---- Panel B: the clips differ in capture quality, not standoff ---------
    ax = axes[1]
    _style(ax)
    ax.grid(True, axis="y", color=NEUTRAL, alpha=0.35, linewidth=0.7)
    data = [df.loc[df["clip"] == c, "sharpness"].dropna().to_numpy() for c in clips]
    parts = ax.violinplot(data, positions=x, widths=0.72, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(NEUTRAL)
        body.set_alpha(0.45)
        body.set_edgecolor(INK_FAINT)
        body.set_linewidth(0.7)
    for xi, vals in zip(x, data):
        med = float(np.median(vals))
        ax.plot([xi - 0.24, xi + 0.24], [med, med], color=INK, lw=2.2, zorder=4)
        ax.text(xi, med + 18, f"{med:.0f}", ha="center", va="bottom",
                fontsize=8, color=INK, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[c]}\n{c[:10]}" for c in clips],
                       fontsize=8.5, color=INK_SOFT)
    ax.set_ylabel("capture sharpness  (variance of Laplacian)",
                  fontsize=9.5, color=INK_SOFT)
    ax.set_title("B · What separates them is capture quality",
                 fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")
    worst = max(clips, key=lambda c: df.loc[df["clip"] == c, "fp_det_v1"].mean()) \
        if "fp_det_v1" in df.columns else None
    if worst:
        ax.annotate("the clip that breaks\nthe detector",
                    xy=(clips.index(worst), float(np.median(
                        df.loc[df["clip"] == worst, "sharpness"]))),
                    xytext=(clips.index(worst) - 0.15, 470),
                    fontsize=8.5, color=INK_SOFT, ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK_FAINT, lw=0.9))

    # ---- Panel C: opposite failure modes ------------------------------------
    ax = axes[2]
    _style(ax)
    edges = np.quantile(df["sharpness"].dropna(), np.linspace(0, 1, 6))
    centres, series = [], {m: [] for m in models}
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        sel = df[(df["sharpness"] >= lo_e) & (df["sharpness"] <= hi_e)]
        if len(sel) < 20:
            continue
        centres.append(float(sel["sharpness"].mean()))
        for m in models:
            series[m].append(float(sel[f"fp_{m}"].mean()))
    for m in models:
        ax.plot(centres, series[m], marker="o", markersize=7, lw=2.2,
                color=MODEL_COLOURS[m], label=MODEL_LABELS[m], zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
    ax.margins(x=0.10)
    # Stated precisely: seg v3 is U-shaped (it fires at BOTH extremes), so the
    # genuine opposition is detector vs seg-gpu, and the caption says only that.
    ax.text(0.5, 0.99,
            "detector fires on SHARP frames (r = +0.60)\n"
            "seg-gpu fires on DEGRADED ones (r = −0.11)\n"
            "seg v3 fires at both extremes (r = +0.04, n.s.)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3,
            color=INK_SOFT, style="italic", linespacing=1.6)
    ax.set_xlabel("capture sharpness", fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel("frames with ≥1 false alarm", fontsize=9.5, color=INK_SOFT)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("C · Detector and seg-gpu fail in opposite regimes",
                 fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=MODEL_COLOURS[m], label=MODEL_LABELS[m])
               for m in models]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=len(models),
               loc="upper left", bbox_to_anchor=(0.052, 0.845),
               labelcolor=INK_SOFT, handletextpad=0.5, columnspacing=2.2)

    fig.suptitle(
        "What drives false alarms on real undamaged net — 638 frames, 4 clips, 2 days",
        fontsize=13.5, color=INK, fontweight="bold", x=0.055, ha="left", y=0.955)
    fig.text(0.055, 0.905,
             "Every SOLAQUA frame shows undamaged net, so every detection is a known false positive — no annotation needed.  "
             "The pre-registered hypothesis that standoff distance\nexplained the different-day gap was tested and REJECTED: "
             "three clips flown at near-identical standoff differ by 0 → 31%.  Recall on real damage remains unmeasured.",
             fontsize=8.8, color=INK_SOFT, ha="left", va="top", linespacing=1.5)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default=FRAME_CONDITIONS)
    ap.add_argument("--out", default="docs/images/what_drives_false_alarms.png")
    args = ap.parse_args()

    import pandas as pd
    path = Path(args.conditions)
    if not path.exists():
        raise SystemExit(f"{path} not found — run scripts/analyze_operating_envelope.py first.")
    out = build(pd.read_parquet(path), Path(args.out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

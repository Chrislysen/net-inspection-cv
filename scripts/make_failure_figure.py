"""Build the README figure showing WHAT the detector actually fires on.

The statistical result — one clip produces 31% false alarms while two clips flown
at the same standoff on the same day produce none — says a scene effect exists.
It does not say what the scene effect *is*. Looking at the frames answers that,
and the answer is visible without any analysis: the clip that breaks the detector
is the one rigged with **fiducial calibration markers and their mooring cords**.
The clean clips are plain net.

Drawing the boxes sharpens it further: the detector does not fire on the markers.
It fires on the **thin bright cords** rigged around them — elongated high-contrast
structures that resemble the synthetic tears it was trained on. The clean clip
also contains hardware (floats, smaller markers) at greater distance and produces
nothing, so the trigger is not "equipment in frame" but a specific shape at a
specific scale.

So "capture sharpness" (r = +0.60 with false alarms) was never the cause — it was
a proxy for *this frame contains thin bright foreign structure*.

Stated honestly: this is an observation from the imagery, on one clip. It
explains the measured correlation and is consistent with every clip-level rate,
but it is not a controlled experiment, and no attempt is made here to quantify
"cord present" as a variable. Doing so properly would need the frames labelled
for rigging, which is exactly the kind of annotation this project does not have.

Run: ``python scripts/make_failure_figure.py``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect.inference import NetInspector
from netinspect.utils import ensure_dir, get_logger, read_image

LOGGER = get_logger()

FRAME_CONDITIONS = "reports/results/operating_envelope/frame_conditions.parquet"
WEIGHTS = "models/yolo_damage_v1.pt"

SURFACE, INK, INK_SOFT = "#fcfcfb", "#0b0b0b", "#52514e"
ALARM = "#eb6834"
CLEAN = "#1baf7a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default=FRAME_CONDITIONS)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="docs/images/what_the_detector_fires_on.jpg")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_parquet(args.conditions)
    worst = df.groupby("clip")["fp_det_v1"].mean().idxmax()
    clean_pool = df[(df["clip"] != worst) & (df["clip"].str.startswith(worst[:10]))]
    clean = clean_pool.groupby("clip")["fp_det_v1"].mean().idxmin()
    LOGGER.info("alarm clip %s vs clean clip %s", worst, clean)

    # Frames the detector actually fired on, and clean frames for contrast.
    fires = df[(df["clip"] == worst) & (df["fp_det_v1"] == 1)]
    fires = fires.sort_values("ndet_det_v1", ascending=False).head(args.n)
    quiet = df[df["clip"] == clean].iloc[
        np.linspace(0, len(df[df["clip"] == clean]) - 1, args.n).astype(int)]

    insp = NetInspector(yolo_weights=args.weights)

    fig, axes = plt.subplots(2, args.n, figsize=(4.05 * args.n, 6.5),
                             facecolor=SURFACE)
    fig.subplots_adjust(top=0.76, bottom=0.05, hspace=0.14, wspace=0.03,
                        left=0.012, right=0.988)

    for row, (frames, colour, tag) in enumerate(
            ((fires, ALARM, worst), (quiet, CLEAN, clean))):
        for col in range(args.n):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col >= len(frames):
                ax.axis("off")
                continue
            r = frames.iloc[col]
            img = read_image(r["path"])
            ax.imshow(img)
            boxes = insp.predict(img, method="yolo", conf=args.conf).boxes
            for b in boxes:
                ax.add_patch(mpatches.Rectangle(
                    (b.x1, b.y1), b.width, b.height, fill=False,
                    edgecolor=colour, linewidth=2.4))
                ax.text(b.x1, b.y1 - 6, f"{b.score:.2f}", color=colour,
                        fontsize=8, fontweight="bold")
            ax.set_title(f"{len(boxes)} detection{'s' if len(boxes) != 1 else ''}"
                         f"   ·   standoff {r['standoff']:.2f} m",
                         fontsize=9, color=colour if boxes else INK_SOFT, pad=5)

        rate = df.loc[df["clip"] == tag, "fp_det_v1"].mean()
        axes[row, 0].set_ylabel(f"clip {tag[-8:]}\n{rate:.0%} false-alarm rate",
                                fontsize=10.5, color=colour, fontweight="bold",
                                labelpad=10)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for sp in axes[row, 0].spines.values():
            sp.set_visible(False)

    fig.suptitle("What the detector actually fires on — both clips are UNDAMAGED net",
                 fontsize=14, color=INK, fontweight="bold", x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.915,
             "Same day, same site, same camera, near-identical standoff — and every box below is a FALSE POSITIVE, because there is no damage in any of these frames.\n"
             "The boxes do not land on the calibration markers. They land on the thin bright mooring cords rigged around them: elongated high-contrast structures that\n"
             "resemble the synthetic tears this detector was trained on. Note the bottom row also carries hardware — floats and markers, further away — and fires nothing,\n"
             "so the trigger is not 'equipment in frame'. This is what the measured sharpness correlation (r = +0.60) was standing in for.",
             fontsize=9.2, color=INK_SOFT, ha="left", va="top", linespacing=1.55)

    out = Path(args.out)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=125, facecolor=SURFACE, bbox_inches="tight",
                pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

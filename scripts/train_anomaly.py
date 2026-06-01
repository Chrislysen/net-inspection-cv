"""Fit the normal-net anomaly model on a set of (normal) frames.

Use SOLAQUA frames — they are undamaged net, i.e. exactly the "normal" data this
one-class model needs.

Example
-------
    python scripts/train_anomaly.py --images data/processed/solaqua_frames --out outputs/anomaly/model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from netinspect.anomaly import AnomalyConfig, fit
from netinspect.utils import list_images, read_image, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Directory of normal frames")
    ap.add_argument("--out", required=True, help="Model path prefix (writes .npz)")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--threshold-percentile", type=float, default=99.0)
    ap.add_argument("--holdout", type=int, default=0,
                    help="Reserve the last K frames from training (sanity check)")
    args = ap.parse_args()

    images = list_images(args.images)
    if not images:
        print(f"No images in {args.images}. Run scripts/fetch_solaqua.py first.")
        return
    train_imgs = images[:-args.holdout] if args.holdout else images
    if not train_imgs:
        print("All frames held out; reduce --holdout.")
        return

    cfg = AnomalyConfig(resize=args.resize, grid=args.grid,
                        threshold_percentile=args.threshold_percentile)
    model = fit([read_image(p) for p in train_imgs], cfg)
    out = Path(args.out)
    model.save(out)
    write_json({"train_images": len(train_imgs), "threshold": model.threshold,
                **model.train_stats}, out.with_name(out.name + "_trainstats.json"))
    print(f"Trained on {len(train_imgs)} frames. Threshold={model.threshold:.3f}")
    print(f"Saved model to {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

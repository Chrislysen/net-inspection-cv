"""Fit a PatchCore deep-feature anomaly model on normal net frames.

PatchCore needs only *normal* frames (no damage labels). It uses a pretrained
torchvision backbone, so it is a much stronger anomaly localiser than the
hand-crafted model in `train_anomaly.py` (F1 ~0.78 vs ~0.12 on the composite
test set).

Threshold: by default ``threshold_factor x median`` training distance. For best
results, calibrate the threshold on a small *labelled* validation set (e.g. the
composite val split) and pass it via ``--threshold``.

Example
-------
    python scripts/train_patchcore.py --images data/processed/solaqua_frames_dense \\
        --out models/patchcore_normal_net
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from netinspect.patchcore import PatchCoreConfig, fit
from netinspect.utils import list_images, read_image, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Directory of normal frames")
    ap.add_argument("--out", required=True, help="Model path prefix (.npz)")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--coreset", type=int, default=4000)
    ap.add_argument("--threshold-factor", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Absolute anomaly threshold (overrides factor; calibrate on labelled val)")
    ap.add_argument("--max-frames", type=int, default=150)
    args = ap.parse_args()

    images = list_images(args.images)[: args.max_frames]
    if not images:
        print(f"No frames in {args.images}.")
        return
    cfg = PatchCoreConfig(backbone=args.backbone, coreset_size=args.coreset,
                          threshold_factor=args.threshold_factor)
    model = fit([read_image(p) for p in images], cfg)
    if args.threshold is not None:
        model.threshold = float(args.threshold)
    out = Path(args.out)
    model.save(out)
    write_json({"train_frames": len(images), "threshold": model.threshold,
                **model.train_stats}, out.with_name(out.name + "_trainstats.json"))
    print(f"Fitted PatchCore on {len(images)} frames; bank={model.train_stats['bank']}, "
          f"threshold={model.threshold:.3f}")
    print(f"Saved {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

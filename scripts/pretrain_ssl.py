"""Self-supervised (SimCLR) pretraining of a ResNet18 backbone on unlabelled SOLAQUA.

The deferred "domain SSL pretraining" experiment, implemented from scratch (no SSL
library): instead of an ImageNet-supervised or off-the-shelf-DINOv2 backbone, learn
features *directly from the unlabelled SOLAQUA net frames* by contrastive learning
(two augmented views of each frame pulled together, others pushed apart — NT-Xent),
then drop the backbone into the PatchCore anomaly detector and re-measure.

Honesty
-------
* It pretrains on a few hundred frames from **one site** — a *proof of concept* of
  domain pretraining, not large-scale SSL (which wants 10^5-10^6 images).
* Pretrain only on the **training-day** clips; never the held-out different-day clip,
  or the OOD test is no longer OOD.
* Evaluation is still on synthetic/undamaged proxies — this can change *which
  features* the detector uses, not whether real-damage accuracy is validated.

Example
-------
    python scripts/pretrain_ssl.py --frames data/processed/solaqua_frames_dense \\
        data/processed/solaqua_bag2 data/processed/solaqua_bag3 \\
        --epochs 200 --batch 128 --device cuda --out models/ssl_resnet18_solaqua.pt
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect.ssl_pretrain import SimCLRConfig, pretrain
from netinspect.utils import get_logger, list_images

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, nargs="+", help="Frame dirs (training days only)")
    ap.add_argument("--out", required=True, help="Output backbone weights (.pt)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    frames: list = []
    for d in args.frames:
        frames.extend(list_images(d))
    if not frames:
        print(f"No frames found in {args.frames}. Run scripts/fetch_solaqua.py first.")
        return
    LOGGER.info("SimCLR pretraining on %d unlabelled frames (NO held-out day).", len(frames))
    cfg = SimCLRConfig(epochs=args.epochs, batch=args.batch, img_size=args.img,
                       lr=args.lr, temperature=args.temperature,
                       device=args.device, workers=args.workers)
    pretrain(frames, args.out, cfg)
    print(f"\nSaved SOLAQUA-pretrained ResNet18 backbone -> {args.out}")
    print("Use it in PatchCore: scripts/train_patchcore.py --backbone-weights " + args.out)


if __name__ == "__main__":
    main()

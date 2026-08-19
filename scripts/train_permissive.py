"""Train the AGPL-free detector on your own prepared dataset.

The shipped YOLO weights derive from Ultralytics (AGPL-3.0), which is viral over
a network and is where most corporate legal reviews stop. This trains a
torchvision detector (BSD-3-Clause) instead, so the resulting artifact carries no
Ultralytics obligation.

    netinspect onboard ./my_footage --out data/mysite
    python scripts/train_permissive.py --data data/mysite --epochs 30
    netinspect gate --data data/mysite --method permissive \\
        --weights models/permissive_v1.pt

Accepts any dataset laid out by ``netinspect onboard``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect import dataset as D
from netinspect.permissive_baseline import ARCHITECTURES, DEFAULT_ARCH, PermissiveConfig, train
from netinspect.utils import get_logger

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="dataset directory from `netinspect onboard`")
    ap.add_argument("--split", default="train")
    ap.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(ARCHITECTURES))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--water-augment", type=float, default=0.0,
                    help="fraction of frames degraded through the Jerlov water "
                         "model (0 = off)")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    ap.add_argument("--out", default="models/permissive_v1.pt")
    args = ap.parse_args()

    split_dir = Path(args.data) / "images" / args.split
    if not split_dir.exists():
        split_dir = Path(args.data)          # allow a bare folder too
    samples = D.load_dataset(split_dir, fmt="yolo")
    labelled = sum(1 for s in samples if s.boxes)
    print(f"{len(samples)} frames ({labelled} labelled, "
          f"{len(samples) - labelled} clean) from {split_dir}")

    cfg = PermissiveConfig(arch=args.arch, epochs=args.epochs,
                           batch_size=args.batch_size, lr=args.lr,
                           water_augment=args.water_augment)
    summary = train(samples, cfg, out_path=args.out, device=args.device)

    print(f"\nwrote {summary['weights']}")
    print(f"  arch        {summary['arch']}")
    print(f"  final loss  {summary['final_loss']}")
    print(f"  licence     {summary['licence'].splitlines()[0]}")
    print(f"\nnext:  netinspect gate --data {args.data} --method permissive "
          f"--weights {summary['weights']}")


if __name__ == "__main__":
    main()

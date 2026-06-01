"""Generate the synthetic placeholder dataset (PIPELINE TESTING ONLY).

This writes procedurally generated net-like images with injected artificial
"damage" and YOLO ground-truth labels. It exists to exercise the code path
end-to-end without real data.

  *** Synthetic data only verifies that the pipeline works. ***
  *** It does NOT prove real aquaculture performance.        ***

Example
-------
    python scripts/make_demo_data.py --out data/sample --n 8
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect.synthetic import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sample", help="Output dataset root")
    ap.add_argument("--n", type=int, default=8, help="Number of images")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (reproducible)")
    args = ap.parse_args()

    info = generate_dataset(args.out, n_images=args.n, seed=args.seed)
    print(f"Generated {info['n_images']} synthetic images "
          f"({info['n_damaged']} damaged) in {info['out_dir']}")
    print("Layout: <out>/images/*.jpg, <out>/labels/*.txt")
    print("\n*** SYNTHETIC PLACEHOLDER DATA — not representative of real conditions. ***")


if __name__ == "__main__":
    main()

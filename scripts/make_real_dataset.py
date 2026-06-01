"""Build a YOLO dataset by compositing synthetic damage onto REAL net frames.

Backgrounds are real SOLAQUA frames; damage is synthetic (see compose.py).
This produces a realistic, fully-labelled train/val/test set for training and
for quantitatively comparing methods — clearly NOT real damage.

Example
-------
    python scripts/make_real_dataset.py --frames data/processed/solaqua_frames_dense \\
        --out data/processed/real_composite --seg
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401
from netinspect.compose import ComposeConfig, build_dataset
from netinspect.utils import list_images, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="Directory of real frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", action="store_true", help="Write segmentation polygons")
    ap.add_argument("--damaged-fraction", type=float, default=0.85)
    ap.add_argument("--max-damage", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--multiclass", action="store_true",
                    help="Keep hole/tear subtypes (default: single 'damage' class)")
    args = ap.parse_args()

    frames = list_images(args.frames)
    if not frames:
        print(f"No frames in {args.frames}. Run scripts/fetch_solaqua.py first.")
        return

    cfg = ComposeConfig(single_class=not args.multiclass)
    info = build_dataset(frames, args.out, damaged_fraction=args.damaged_fraction,
                         max_damage_per_image=args.max_damage, seg=args.seg,
                         seed=args.seed, cfg=cfg)
    write_json(info, f"{args.out}/dataset_info.json")
    print(f"Built dataset at {info['out_dir']} from {info['num_frames']} real frames:")
    for split, c in info["splits"].items():
        print(f"  {split:5s}: {c['images']} images, {c['damaged']} damaged, "
              f"{c['instances']} damage instances")
    print(f"\nLabels: {'segmentation polygons' if args.seg else 'detection boxes'}; "
          f"classes: {'hole/tear/...' if args.multiclass else 'single damage'}")
    print("\nNOTE: synthetic damage on REAL backgrounds. Not real damage; validate "
          "on real labelled damage before any reliability claim.")


if __name__ == "__main__":
    main()

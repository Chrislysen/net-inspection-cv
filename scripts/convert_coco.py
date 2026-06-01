"""Convert a COCO-format dataset to YOLO for use in this pipeline.

This is the drop-in slot for real labelled data (which usually arrives as COCO).

Honest note for public underwater datasets (SeaClear / TrashCan / Trash-ICRA19):
they are marine DEBRIS, not net damage — useful for transfer/pretraining and as a
real-image smoke test of this converter, NOT as a damage proxy. Verify each
dataset's licence before downloading/using.

Examples
--------
    python scripts/convert_coco.py --coco data/seaclear/annotations.json \\
        --images data/seaclear/images --out data/processed/seaclear --single-class
    python scripts/convert_coco.py --coco anns.json --images imgs --out out --seg
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401
from netinspect.coco import convert_coco_to_yolo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="COCO annotation JSON")
    ap.add_argument("--images", required=True, help="Image directory")
    ap.add_argument("--out", required=True, help="Output YOLO dataset dir")
    ap.add_argument("--seg", action="store_true", help="Write segmentation polygons")
    ap.add_argument("--single-class", action="store_true",
                    help="Collapse all annotations to one class (index 0)")
    ap.add_argument("--no-copy", action="store_true", help="Do not copy images")
    args = ap.parse_args()

    res = convert_coco_to_yolo(args.coco, args.images, args.out, segmentation=args.seg,
                               single_class=args.single_class, copy_images=not args.no_copy)
    print(f"Converted -> {res.out_dir}")
    print(f"  images:    {res.num_images}")
    print(f"  labelled:  {res.num_labels}")
    print(f"  instances: {res.num_instances}")
    print(f"  classes:   {res.class_names}")
    if res.skipped:
        print(f"  skipped:   {len(res.skipped)} (missing image or size)")
    print(f"\nTrain: python scripts/train_yolo.py --data {args.out}/dataset.yaml --epochs 50")
    print("Note: verify the dataset's licence; public debris sets are NOT a net-damage proxy.")


if __name__ == "__main__":
    main()

"""Measure the polarity filter: how many false alarms go, and what recall costs.

``netinspect.polarity`` rejects detections whose interior is brighter than the
net around them. The rationale is physical — a hole shows unlit water and is
dark; a rope is an object in front of the net and is bright — and it came out of
measuring the false alarms rather than guessing at them
(``scripts/analyse_false_alarms.py``).

This script decides whether it is worth switching on, by sweeping the threshold
and reporting BOTH sides of the trade at every point:

* false-alarm frame rate on all four real undamaged clips (557 frames), where
  every detection is wrong by construction and no labels are needed;
* recall on the labelled composite test split, where the cost shows up.

    python scripts/eval_polarity_filter.py

The result to be sceptical of
-----------------------------
The damage in the labelled split is **synthetic**, and the generator paints it
dark by construction. A darkness filter evaluated against it is being graded on
an assumption it shares, so the recall column here is optimistic in a way the
false-alarm column is not — the false alarms are real net. Read the recall
number as "this does not obviously break the detector", not as "this is safe on
real damage". That question needs real labelled damage, which the project does
not have.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401

from netinspect import acceptance as A
from netinspect import dataset as D
from netinspect import polarity as POL
from netinspect.inference import NetInspector
from netinspect.utils import BBox, ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()
REPO = _common.REPO_ROOT

CLIPS = {
    "bag1": "data/processed/solaqua_frames",
    "bag2": "data/processed/solaqua_bag2",
    "bag3": "data/processed/solaqua_bag3",
    "diffday": "data/processed/solaqua_diffday",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--method", default="yolo")
    ap.add_argument("--labelled", default="data/processed/real_composite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[-1e9, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
    ap.add_argument("--out", default="reports/results/polarity_filter")
    args = ap.parse_args()

    insp = NetInspector(yolo_weights=args.weights)

    # Infer ONCE and cache the raw boxes; the filter is a pure post-step, so
    # re-running the detector per threshold would only add noise and minutes.
    LOGGER.info("running the detector once over every frame")
    clean: dict[str, list] = {}
    for clip, rel in CLIPS.items():
        clean[clip] = []
        for p in list_images(REPO / rel):
            img = read_image(p)
            clean[clip].append((img, insp.predict(img, method=args.method,
                                                  conf=args.conf).boxes))

    samples = D.load_dataset(Path(args.labelled) / "images" / args.split, fmt="yolo")
    labelled = []
    for s in samples:
        img = read_image(s.image)
        h, w = img.shape[:2]
        gt = [BBox(x1=b.x1 * w, y1=b.y1 * h, x2=b.x2 * w, y2=b.y2 * h,
                   score=1.0, class_name="damage") for b in s.boxes]
        labelled.append((s.image.name, img, gt,
                         insp.predict(img, method=args.method, conf=args.conf).boxes))

    rows = []
    for thr in args.thresholds:
        per_clip, tot_a, tot_f = {}, 0, 0
        for clip, frames in clean.items():
            alarming = 0
            for img, boxes in frames:
                kept = POL.filter_detections(img, boxes, min_contrast=thr)
                if kept:
                    alarming += 1
            per_clip[clip] = {"frames": len(frames), "alarming": alarming,
                              "rate": alarming / max(1, len(frames))}
            tot_a += alarming
            tot_f += len(frames)

        preds = {name: POL.filter_detections(img, boxes, min_contrast=thr)
                 for name, img, _gt, boxes in labelled}
        gts = {name: gt for name, _img, gt, _b in labelled}
        rec = A.measure(preds, gts, conf=args.conf)

        rows.append({"min_contrast": thr, "per_clip": per_clip,
                     "overall_rate": tot_a / max(1, tot_f),
                     "alarming_frames": tot_a, "frames": tot_f,
                     "recall": rec.get("recall"), "precision": rec.get("precision")})

    base = rows[0]
    print(f"\nPolarity filter sweep · {base['frames']} real undamaged frames · "
          f"{len(samples)} labelled frames · conf>={args.conf}\n")
    hdr = f"{'min_contrast':>12s} {'bag1':>7s} {'bag2':>7s} {'bag3':>7s} {'diffday':>8s} " \
          f"{'OVERALL':>9s} {'recall':>8s}"
    print(hdr)
    for r in rows:
        pc = r["per_clip"]
        thr = "off" if r["min_contrast"] < -1e8 else f"{r['min_contrast']:.0f}"
        rec = "n/a" if r["recall"] is None else f"{r['recall']:.0%}"
        print(f"{thr:>12s} {pc['bag1']['rate']:>6.1%} {pc['bag2']['rate']:>7.1%} "
              f"{pc['bag3']['rate']:>7.1%} {pc['diffday']['rate']:>8.1%} "
              f"{r['overall_rate']:>9.1%} {rec:>8s}")

    kept = [r for r in rows if r["recall"] is not None
            and base["recall"] is not None and r["recall"] >= base["recall"]]
    best = min(kept, key=lambda r: r["overall_rate"]) if kept else None
    if best is not None and best["min_contrast"] > -1e8:
        print(f"\nAt min_contrast={best['min_contrast']:.0f} the overall false-alarm rate "
              f"falls {base['overall_rate']:.1%} -> {best['overall_rate']:.1%} "
              f"with recall unchanged at {best['recall']:.0%}.")
    print("\nThe recall column is measured on SYNTHETIC damage, which the generator "
          "paints dark by\nconstruction — the same assumption this filter makes. It is "
          "evidence that the filter does\nnot obviously break the detector, NOT evidence "
          "that it is safe on real damage.")

    out = ensure_dir(args.out)
    write_json({"config": vars(args), "sweep": rows,
                "caveat": ("Recall is measured on synthetic damage that is dark by "
                           "construction; the filter assumes darkness. Transfer to real "
                           "damage is unvalidated.")},
               Path(out) / "polarity_filter.json")
    print(f"\nwrote {out}/polarity_filter.json")


if __name__ == "__main__":
    main()

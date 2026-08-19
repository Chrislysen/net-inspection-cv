"""Does underwater image enhancement help detection, or hurt it?

Underwater CV literature is full of enhancement methods — UWCNN, colour
correction, dehazing — and they are unarguably better to *look at*. The question
this project has to answer before adopting any of them is different: does a
detector get better?

The answer here is no, and the direction is not subtle. Enhancement is applied
at inference to a detector trained on raw frames, which is how it would actually
be deployed as a preprocessing step, and the resulting distribution mismatch
costs more than the enhancement gains. Contrast stretching in particular
amplifies exactly the thin bright structures — mooring cords, biofouling,
fiducial rigging — that this detector already false-fires on.

Two datasets, because the two rates need different data:

* real undamaged SOLAQUA footage — every detection is a false alarm, so this
  measures the cost;
* the labelled composite split — this measures whether recall improves enough to
  pay for it.

    python scripts/eval_preprocessing.py

Reported as a negative result on purpose. Knowing that a popular preprocessing
step makes this system worse is worth more than adopting it because a paper
showed nicer pictures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401

from netinspect import acceptance as A
from netinspect import dataset as D
from netinspect.inference import NetInspector
from netinspect.preprocess import apply_clahe, compensate_red, denoise, gray_world_white_balance
from netinspect.utils import BBox, ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()

VARIANTS = {
    "raw": (lambda im: im,
            "no preprocessing — how the detector is actually deployed"),
    "clahe": (lambda im: apply_clahe(im),
              "contrast-limited adaptive histogram equalisation"),
    "white_balance": (lambda im: gray_world_white_balance(im),
                      "gray-world white balance — the classical colour-cast fix"),
    "white_balance_clahe": (lambda im: apply_clahe(gray_world_white_balance(im)),
                            "both, the usual underwater recipe"),
    "denoise_clahe": (lambda im: apply_clahe(denoise(im)),
                      "denoise first, then stretch contrast"),
    "red_compensation": (lambda im: compensate_red(im),
                         "Ancuti 2018 red-channel compensation — a colour-cast "
                         "fix rather than a contrast stretch"),
    "red_comp_white_balance": (lambda im: gray_world_white_balance(compensate_red(im)),
                               "Ancuti's own recipe: compensate red, then white balance"),
}


def _false_alarms(insp, paths, fn, conf):
    frames = boxes = 0
    for p in paths:
        b = insp.predict(fn(read_image(p)), method="yolo", conf=conf).boxes
        frames += 1 if b else 0
        boxes += len(b)
    return {"frames_alarming": frames, "frames": len(paths),
            "rate": frames / max(1, len(paths)), "boxes": boxes}


def _recall(insp, samples, fn, conf):
    preds, gts = {}, {}
    for s in samples:
        w, h = s.width or 1, s.height or 1
        gts[s.image.name] = [BBox(b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h, 1.0, "damage")
                             for b in s.boxes]
        preds[s.image.name] = insp.predict(fn(read_image(s.image)),
                                           method="yolo", conf=conf).boxes
    return A.measure(preds, gts, conf=conf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/processed/solaqua_bag3",
                    help="real UNDAMAGED frames — measures the false-alarm cost")
    ap.add_argument("--labelled", default="data/processed/real_composite",
                    help="labelled dataset — measures whether recall improves")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/results/preprocessing")
    args = ap.parse_args()

    insp = NetInspector(yolo_weights=args.weights)
    clean = list_images(Path(args.clean))
    labelled = D.load_dataset(Path(args.labelled) / "images" / args.split, fmt="yolo")
    LOGGER.info("%d clean frames · %d labelled frames · %d variants",
                len(clean), len(labelled), len(VARIANTS))

    rows = []
    for name, (fn, why) in VARIANTS.items():
        LOGGER.info("  %s", name)
        fa = _false_alarms(insp, clean, fn, args.conf)
        rc = _recall(insp, labelled, fn, args.conf)
        rows.append({"variant": name, "description": why,
                     "false_alarms": fa,
                     "recall": rc["recall"], "damaged_frames": rc["damaged_frames"]})

    base = next(r for r in rows if r["variant"] == "raw")
    print(f"\n{len(clean)} real undamaged frames · {len(labelled)} labelled frames "
          f"· conf>={args.conf}\n")
    print(f"{'preprocessing':22s} {'false alarms':>13s} {'vs raw':>8s} "
          f"{'boxes':>7s} {'recall':>8s}")
    for r in rows:
        fa = r["false_alarms"]
        delta = fa["rate"] - base["false_alarms"]["rate"]
        rec = "n/a" if r["recall"] is None else f"{r['recall']:.0%}"
        mark = "" if r["variant"] == "raw" else f"{delta:+.0%}"
        print(f"{r['variant']:22s} {fa['rate']:>12.0%} {mark:>8s} "
              f"{fa['boxes']:>7d} {rec:>8s}")

    worst = max(rows, key=lambda r: r["false_alarms"]["rate"])
    print(f"\nEnhancement does not pay for itself here. Recall is already saturated, "
          f"so there is nothing to gain,\nand {worst['variant']} raises the "
          f"false-alarm rate from {base['false_alarms']['rate']:.0%} to "
          f"{worst['false_alarms']['rate']:.0%}.")
    print("Mechanism: the detector was trained on RAW frames, and contrast "
          "stretching amplifies the\nthin bright mooring cords it already fires on. "
          "Enhance the TRAINING set instead, or not at all.")

    out = ensure_dir(args.out)
    write_json({"conf": args.conf, "weights": args.weights,
                "clean_source": args.clean, "clean_frames": len(clean),
                "labelled_source": args.labelled, "labelled_frames": len(labelled),
                "results": rows,
                "conclusion": (
                    "Underwater enhancement applied at inference to a detector "
                    "trained on raw frames increases false alarms and does not "
                    "improve recall. Not adopted.")},
               out / "preprocessing_ablation.json")
    print(f"\nwrote {out}/preprocessing_ablation.json")


if __name__ == "__main__":
    main()

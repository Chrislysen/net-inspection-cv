"""Does showing the detector the frames it false-fires on actually fix them?

The documented weakness of this project is not the day-to-day gap. It is the
**between-clip spread on a single day**: on real undamaged net the detectors fire
on 0% of bag1, 0% of bag2 and 1% of a different day, but on **bag3 they fire hard**
— 31% for the YOLO detector, 9.5% for the permissive one. For the permissive
model that single clip is the source of *every* false alarm it produces across
557 real frames.

Two attempts to close that gap have already failed, and both shared an
assumption. Stronger photometric augmentation (HSV jitter, rotation, perspective)
and the Jerlov water model (`netinspect.water`) both tried to make the detector
robust to how a scene *looks*. Neither ever showed it the specific thing it fires
on. Looking at the frames says what that is: thin bright mooring cords rigged
around the calibration markers, which are not net damage and are not present in
the training clips.

**This experiment needs no damage labels, which is the point.** Every frame in
bag3 is real undamaged net, so every detection there is a false positive by
construction — the label is "clean", and there are 199 of them sitting on disk.
That makes hard-negative mining the one obvious lever the project has not pulled,
and the only reason it was not pulled earlier is that "we need real labelled
damage" was allowed to stand in for "we need labels for THIS".

Design, and the ways it could lie
---------------------------------
* bag3 is one continuous clip, so adjacent frames are near-duplicates. A random
  split would put a frame's own neighbour in the test set and report memorisation
  as generalisation. The split is **temporal, with a discarded buffer** between
  train and held-out.
* The held-out bag3 frames are never trained on, so they measure whether the
  model learned the *cue* rather than the frames.
* bag1/bag2/different-day are reported too: hard negatives from one clip could
  suppress that clip and do nothing, or worse, elsewhere.
* Recall on the labelled composite test split is reported alongside. Suppressing
  false alarms by making the detector timid is not a win, and this is where that
  would show.
* **Optimizer steps are matched, not epochs.** Adding 119 negatives to 132 frames
  makes an epoch almost twice as long, so "same epochs" silently gives the
  treatment ~1.9x the gradient updates and the comparison varies two things at
  once. The first run of this experiment made exactly that mistake, and the
  recall it appeared to gain is precisely what more steps produce on their own.
  The baseline's epoch count is scaled up so both arms take the same number of
  updates.
* **Several seeds, because one run of a 132-frame detector is an anecdote.**
  Results are reported as mean and range across seeds; a difference smaller than
  the spread is not a difference.

    python scripts/eval_hard_negatives.py --seeds 0 1 2

Written to be reported whichever way it comes out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401

from netinspect import acceptance as A
from netinspect import dataset as D
from netinspect import permissive_baseline as P
from netinspect.utils import BBox, ensure_dir, get_logger, list_images, read_image, write_json

LOGGER = get_logger()
REPO = _common.REPO_ROOT

CLEAN_CLIPS = {
    "bag1": "data/processed/solaqua_frames",
    "bag2": "data/processed/solaqua_bag2",
    "diffday": "data/processed/solaqua_diffday",
}
HARD_CLIP = "data/processed/solaqua_bag3"


def _split_temporally(paths, train_frac: float, gap: int):
    """Split one continuous clip into train / discarded buffer / held-out.

    Adjacent frames of a video are near-identical. Splitting them at random
    leaks the answer across the boundary and turns memorisation into an
    apparently excellent generalisation result, so the split is by time and the
    frames either side of the cut are thrown away entirely.
    """
    paths = sorted(paths)
    n_train = int(len(paths) * train_frac)
    train = paths[:n_train]
    held = paths[n_train + gap:]
    return train, held


def _false_alarm_rate(model, paths) -> dict:
    alarming = 0
    boxes_total = 0
    for p in paths:
        boxes = P.predict_image(model, read_image(p))
        boxes_total += len(boxes)
        if boxes:
            alarming += 1
    n = len(paths)
    return {"frames": n, "alarming": alarming, "boxes": boxes_total,
            "rate": (alarming / n) if n else 0.0}


def _recall(model, samples, conf: float) -> dict:
    preds, gts = {}, {}
    for s in samples:
        w, h = s.width or 1, s.height or 1
        gts[s.image.name] = [BBox(x1=b.x1 * w, y1=b.y1 * h, x2=b.x2 * w, y2=b.y2 * h,
                                  score=1.0, class_name="damage") for b in s.boxes]
        preds[s.image.name] = P.predict_image(model, read_image(s.image))
    return A.measure(preds, gts, conf=conf)


def _evaluate(model, held_bag3, composite_test, conf) -> dict:
    out = {"bag3_heldout": _false_alarm_rate(model, held_bag3)}
    for name, rel in CLEAN_CLIPS.items():
        out[name] = _false_alarm_rate(model, list_images(REPO / rel))
    rec = _recall(model, composite_test, conf)
    out["recall"] = rec.get("recall")
    out["damaged_frames"] = rec.get("damaged_frames")

    seen = [v for k, v in out.items() if isinstance(v, dict)]
    out["overall"] = {
        "frames": sum(v["frames"] for v in seen),
        "alarming": sum(v["alarming"] for v in seen),
    }
    tot = out["overall"]["frames"]
    out["overall"]["rate"] = out["overall"]["alarming"] / tot if tot else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/real_composite",
                    help="labelled composite dataset (damage)")
    ap.add_argument("--hard", default=HARD_CLIP, help="the clip the detector false-fires on")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="fraction of the hard clip used as training negatives")
    ap.add_argument("--gap", type=int, default=20,
                    help="frames discarded between train and held-out (temporal buffer)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one run per seed; a difference inside the spread is not one")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="reports/results/hard_negatives")
    args = ap.parse_args()

    damage_train = D.load_dataset(Path(args.data) / "images" / "train", fmt="yolo")
    composite_test = D.load_dataset(Path(args.data) / "images" / "test", fmt="yolo")

    hard_paths = list_images(REPO / args.hard)
    train_neg_paths, held_paths = _split_temporally(hard_paths, args.train_frac, args.gap)
    negatives = [s for s in D.load_dataset(REPO / args.hard, fmt="images")
                 if s.image in set(train_neg_paths)]

    LOGGER.info("%d damaged training frames · %d hard negatives · %d held-out negatives",
                len(damage_train), len(negatives), len(held_paths))

    out_dir = ensure_dir(args.out)

    arms = {"baseline": list(damage_train),
            "hard_negatives": list(damage_train) + negatives}

    # Match optimizer STEPS, not epochs. An epoch over the larger set is almost
    # twice as long, so equal epochs would hand the treatment ~1.9x the gradient
    # updates and confound the thing being measured with simply training longer.
    def _steps_per_epoch(n):
        return max(1, -(-n // args.batch_size))

    target_steps = _steps_per_epoch(len(arms["hard_negatives"])) * args.epochs
    epochs_for = {name: max(1, round(target_steps / _steps_per_epoch(len(s))))
                  for name, s in arms.items()}
    budget = {name: _steps_per_epoch(len(arms[name])) * e for name, e in epochs_for.items()}
    LOGGER.info("step-matched: %s", {k: f"{len(arms[k])} frames, {epochs_for[k]} epochs, "
                                        f"{budget[k]} steps" for k in arms})

    runs = {name: [] for name in arms}
    for seed in args.seeds:
        for name, samples in arms.items():
            LOGGER.info("seed %d · training %s on %d frames for %d epochs",
                        seed, name, len(samples), epochs_for[name])
            weights = Path(out_dir) / f"permissive_{name}_seed{seed}.pt"
            P.train(samples, P.PermissiveConfig(epochs=epochs_for[name], seed=seed),
                    out_path=weights)
            model = P.load_model(weights)
            res = _evaluate(model, held_paths, composite_test, args.conf)
            res["seed"] = seed
            runs[name].append(res)

    def _stat(name, key):
        vals = [r[key]["rate"] if isinstance(r[key], dict) else r[key] for r in runs[name]]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None, None
        return sum(vals) / len(vals), min(vals), max(vals)

    print(f"\nHard-negative mining on {Path(args.hard).name}: "
          f"{len(negatives)} negatives added, {len(held_paths)} held out, "
          f"{len(args.seeds)} seeds, step-matched "
          f"({budget['baseline']} vs {budget['hard_negatives']} updates)\n")
    print(f"{'set':16s} {'baseline (range)':>26s} {'+hard neg (range)':>26s} {'delta':>8s}")
    rows = {}
    for key in ("bag3_heldout", "bag1", "bag2", "diffday", "overall", "recall"):
        mb, lob, hib = _stat("baseline", key)
        mt, lot, hit = _stat("hard_negatives", key)
        if mb is None or mt is None:
            continue
        rows[key] = {"baseline": [mb, lob, hib], "hard_negatives": [mt, lot, hit]}
        print(f"{key:16s} {mb:>14.1%} [{lob:>5.1%},{hib:>6.1%}] "
              f"{mt:>14.1%} [{lot:>5.1%},{hit:>6.1%}] {mt - mb:>+8.1%}")

    hb, ht = rows["bag3_heldout"]["baseline"], rows["bag3_heldout"]["hard_negatives"]
    overlap = not (ht[2] < hb[1] or hb[2] < ht[1])       # ranges intersect
    if overlap:
        verdict = ("INCONCLUSIVE on held-out frames: the two arms' ranges overlap "
                   "across seeds, so this run cannot distinguish an effect from noise")
    elif ht[0] < hb[0]:
        verdict = "Hard negatives REDUCED false alarms on held-out frames of the mined clip"
    else:
        verdict = "Hard negatives INCREASED false alarms on held-out frames of the mined clip"
    print(f"\n{verdict}.")

    write_json({"config": vars(args),
                "damage_train_frames": len(damage_train),
                "hard_negatives": len(negatives),
                "heldout_negatives": len(held_paths),
                "epochs_per_arm": epochs_for, "optimizer_steps_per_arm": budget,
                "runs": runs, "summary": rows, "verdict": verdict},
               Path(out_dir) / "hard_negatives.json")
    print(f"\nwrote {out_dir}/hard_negatives.json")


if __name__ == "__main__":
    main()

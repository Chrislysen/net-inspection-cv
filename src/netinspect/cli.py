"""``netinspect`` — one command for the whole workflow.

Before this, the toolkit was forty-odd scripts under ``scripts/``. Everything
worked and nothing was discoverable, which for anyone outside the author is the
same as it not working. This is the front door, and it is organised around what
someone actually needs to do, in the order they need to do it:

    netinspect doctor                       # is my environment able to run this?
    netinspect onboard  ./my_data --out ds  # ingest my footage, audit it, split it
    netinspect train    --data ds/dataset.yaml
    netinspect calibrate --data ds --weights runs/.../best.pt
    netinspect gate     --data ds --weights runs/.../best.pt   # may this ship?
    netinspect serve                        # the console
    netinspect predict  ./frames

The important one is ``gate``. It exits non-zero when a model does not meet the
operating point written down beforehand, so a deployment pipeline can refuse to
promote it without anyone having to remember to check.

Commands that wrap an existing script delegate to it rather than duplicating its
logic, so there is exactly one implementation of each thing.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Sequence

from .utils import get_logger

LOGGER = get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

from . import __version__ as VERSION

# Commands that are a thin name over an existing script. Their arguments are
# passed through untouched, which is why they bypass argparse entirely in
# main(): argparse.REMAINDER still lets a leading "--flag" be intercepted as an
# unknown top-level option, so `netinspect train --epochs 5` failed before it
# ever reached the script.
WRAPPED = {
    "train": ("train_yolo.py", "train a detector on your prepared dataset"),
    "predict": ("infer.py", "run detection over images, a video, or a ROS bag"),
    "serve": ("serve.py", "start the web console"),
    "live": ("live_inspect.py", "real-time inference on a camera or RTSP feed"),
    "map": ("map_inspection.py", "turn a pass into a net-frame map of defect sites"),
    "report": ("inspection_report.py", "per-pass inspection validity report"),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _delegate(script: str, args: Sequence[str]) -> int:
    """Run one of the repo's scripts with the given arguments.

    Delegation rather than reimplementation: the scripts are the tested
    implementations, and a second copy behind a nicer name is a second thing to
    keep correct.
    """
    path = SCRIPTS / script
    if not path.exists():
        print(f"error: {script} is not available in this installation "
              f"(looked in {SCRIPTS}).\n"
              "The full toolkit lives in the source checkout; install it with "
              "`pip install -e .` from a clone.", file=sys.stderr)
        return 2
    # runpy.run_path does NOT put the script's directory on sys.path, unlike
    # `python scripts/serve.py`. Every one of these scripts opens with
    # `import _common`, which lives in scripts/, so without this line all six
    # wrapped commands died with ModuleNotFoundError — including `netinspect
    # serve`, which is the Dockerfile's CMD.
    argv, path_backup = sys.argv, list(sys.path)
    sys.path.insert(0, str(SCRIPTS))
    sys.argv = [str(path), *args]
    try:
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:                      # argparse --help, or an error
        return int(exc.code or 0)
    finally:
        sys.argv = argv
        sys.path[:] = path_backup


def _load_split(split_dir: Path):
    """Load one split of a prepared dataset, labels included.

    The format is forced to YOLO rather than sniffed. Sniffing looks only inside
    ``images/<split>/``, which by construction contains no label files — they
    live in the sibling ``labels/`` tree — so detection returned "images" and
    every frame silently arrived with no ground truth. The gate then reported
    zero damaged frames and failed closed, which is the right failure but the
    wrong reason.
    """
    from . import dataset as D
    parts = list(split_dir.parts)
    prepared = "images" in parts and (Path(*[
        "labels" if (i == len(parts) - 2 and p == "images") else p
        for i, p in enumerate(parts)])).exists()
    return D.load_dataset(split_dir, fmt="yolo" if prepared else None)


def _to_pixel_boxes(sample) -> list:
    from .utils import BBox
    w, h = sample.width or 1, sample.height or 1
    return [BBox(x1=b.x1 * w, y1=b.y1 * h, x2=b.x2 * w, y2=b.y2 * h,
                 score=1.0, class_name="damage") for b in sample.boxes]


def _run_detector(samples, weights: str | None, method: str, conf: float):
    """Detections for every sample, keyed by image name."""
    from .classical_baseline import ClassicalConfig
    from .inference import NetInspector
    from .utils import read_image

    # A .pt from train_permissive.py carries its own config; route it to the
    # AGPL-free path rather than to Ultralytics.
    kw = ({"permissive_weights": weights} if method == "permissive"
          else {"yolo_weights": weights})
    insp = NetInspector(classical_cfg=ClassicalConfig(), **kw)
    if method not in insp.available_methods():
        raise SystemExit(f"Method {method!r} is unavailable "
                         f"(have: {insp.available_methods()}). Check --weights.")
    preds, gts = {}, {}
    for i, s in enumerate(samples, 1):
        name = s.image.name
        gts[name] = _to_pixel_boxes(s)
        result = insp.predict(read_image(s.image), method=method, conf=conf)
        preds[name] = list(result.boxes)
        if i % 50 == 0:
            LOGGER.info("  scored %d/%d frames", i, len(samples))
    return preds, gts


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    """What is installed, what is missing, and what that stops you doing."""
    from .utils import optional_import

    groups = {
        "core": [("numpy", "arrays"), ("pandas", "tables"), ("PIL", "image IO")],
        "cv": [("cv2", "classical baseline, video, visual odometry"),
               ("matplotlib", "figures"), ("skimage", "some preprocessing"),
               ("sklearn", "metrics and PatchCore")],
        "ml": [("torch", "neural inference"), ("torchvision", "backbones"),
               ("ultralytics", "YOLO detection and segmentation")],
        "data": [("rosbags", "reading ROV .bag files"), ("pyarrow", "parquet telemetry")],
        "serve": [("fastapi", "HTTP service"), ("uvicorn", "server"),
                  ("streamlit", "alternative viewer")],
        "export": [("onnx", "model export"), ("onnxruntime", "portable inference")],
    }
    missing_extras = []
    print(f"netinspect {VERSION}   python {sys.version.split()[0]}   {sys.platform}\n")
    for extra, mods in groups.items():
        rows = []
        for mod, why in mods:
            ok = optional_import(mod) is not None
            rows.append(f"    [{'x' if ok else ' '}] {mod:<14} {why}")
            if not ok and extra != "core":
                missing_extras.append(extra)
        print(f"  {extra}:")
        print("\n".join(rows))
    print()

    models = sorted((REPO_ROOT / "models").glob("*.pt")) if (REPO_ROOT / "models").exists() else []
    print(f"  models: {len(models)} committed weight file(s)")
    for m in models[:6]:
        print(f"    - {m.name}")

    for extra in sorted(set(missing_extras)):
        print(f"\n  to enable '{extra}':  pip install -e \".[{extra}]\"")
    print("\n  Reminder: the shipped weights learned SYNTHETIC damage. Their recall on "
          "real holes is unmeasured — run `netinspect gate` against your own labelled "
          "footage before trusting any of it.")
    return 0


def cmd_onboard(args) -> int:
    """Ingest a customer dataset: detect the format, audit it, split it safely."""
    from . import dataset as D

    root = Path(args.input)
    samples = D.load_dataset(root, fmt=args.format)
    issues = D.audit(samples)
    splits = D.split_samples(samples, ratios=tuple(args.ratios), seed=args.seed,
                             group=not args.no_group)
    dupes = [] if args.skip_duplicate_check else D.find_cross_split_duplicates(splits)
    report = D.health_report(samples, splits, issues, dupes, grouped=not args.no_group)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_health.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + D.summarise(report) + "\n")

    if not report["usable"]:
        print("Refusing to write a dataset with blocking errors. Fix the labels "
              "above and re-run; the full list is in data_health.json.",
              file=sys.stderr)
        return 1

    yaml_path = D.write_yolo(splits, out, class_names=args.classes)
    print(f"wrote {yaml_path}")
    print(f"wrote {out / 'data_health.json'}")
    # The duplicate warning is already in the summary above; repeating it on
    # stderr only made it surface out of order relative to stdout.
    print(f"\nnext:  netinspect train --data {yaml_path}")
    return 0


def cmd_calibrate(args) -> int:
    """Pick the confidence threshold on the customer's own validation split."""
    from . import acceptance as A

    split = Path(args.data) / "images" / args.split
    if not split.exists():
        raise SystemExit(f"No such split: {split}")
    samples = _load_split(split)
    LOGGER.info("Calibrating on %d frames from %s", len(samples), split)
    preds, gts = _run_detector(samples, args.weights, args.method, conf=0.01)
    out = A.choose_threshold(preds, gts, target_false_alarm_rate=args.max_false_alarms)

    print(f"\ntarget false-alarm rate: {args.max_false_alarms:.1%}")
    print(f"{'conf':>6} {'false alarms':>14} {'recall':>8}")
    for r in out["sweep"]:
        fa = "n/a" if r["false_alarm_rate"] is None else f"{r['false_alarm_rate']:.1%}"
        rc = "n/a" if r["recall"] is None else f"{r['recall']:.1%}"
        mark = " <-" if out["chosen"] and r["conf"] == out["chosen"]["conf"] else ""
        print(f"{r['conf']:>6.2f} {fa:>14} {rc:>8}{mark}")

    print("\n" + out["note"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if not out["achievable"]:
        return 1
    print(f"\nchosen threshold: {out['chosen']['conf']:.2f}")
    return 0


def cmd_gate(args) -> int:
    """Decide whether this model may be deployed. Non-zero exit means no."""
    from . import acceptance as A

    split = Path(args.data) / "images" / args.split
    if not split.exists():
        raise SystemExit(f"No such split: {split}")
    # A contract kept in version control beats one retyped on the command line:
    # the flags then only override what the file does not say.
    if args.operating_point:
        import yaml
        text = Path(args.operating_point).read_text(encoding="utf-8")
        loaded = (json.loads(text) if args.operating_point.endswith(".json")
                  else yaml.safe_load(text)) or {}
        op = A.OperatingPoint.from_dict(loaded)
        print(f"operating point from {args.operating_point}")
    else:
        op = A.OperatingPoint(conf=args.conf, iou=args.iou,
                              max_false_alarm_rate=args.max_false_alarms,
                              min_recall=args.min_recall,
                              min_precision=args.min_precision,
                              min_clean_frames=args.min_clean_frames,
                              min_damaged_frames=args.min_damaged_frames)
    samples = _load_split(split)
    LOGGER.info("Gating on %d frames from %s", len(samples), split)
    preds, gts = _run_detector(samples, args.weights, args.method, conf=0.01)
    verdict = A.gate(preds, gts, op)

    print("\n" + verdict.summary() + "\n")
    if args.out:
        A.write_verdict(verdict, args.out)
        print(f"wrote {args.out}")
    return 0 if verdict.passed else 1


def cmd_version(args) -> int:
    print(f"netinspect {VERSION}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="netinspect",
        description="Net-damage inspection toolkit — data in, gated model out.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical first run:\n"
            "  netinspect doctor\n"
            "  netinspect onboard ./my_footage --out data/mysite\n"
            "  netinspect train --data data/mysite/dataset.yaml\n"
            "  netinspect calibrate --data data/mysite --weights runs/detect/train/weights/best.pt\n"
            "  netinspect gate --data data/mysite --weights runs/detect/train/weights/best.pt\n"
        ))
    p.add_argument("--version", action="store_true", help="print the version and exit")
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("doctor", help="check the environment and report what is missing")
    d.set_defaults(func=cmd_doctor)

    o = sub.add_parser("onboard", help="ingest your footage: audit, split, write a dataset")
    o.add_argument("input", help="folder of images (+ YOLO/COCO/VOC labels if you have them)")
    o.add_argument("--out", required=True, help="where to write the prepared dataset")
    o.add_argument("--format", choices=["yolo", "coco", "voc", "images"], default=None,
                   help="override format detection")
    o.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--classes", nargs="+", default=["damage"])
    o.add_argument("--no-group", action="store_true",
                   help="split per image instead of per clip (rarely correct for video)")
    o.add_argument("--skip-duplicate-check", action="store_true",
                   help="skip perceptual hashing (faster on very large sets)")
    o.set_defaults(func=cmd_onboard)

    c = sub.add_parser("calibrate", help="choose the confidence threshold on your own data")
    c.add_argument("--data", required=True, help="dataset directory from `onboard`")
    c.add_argument("--split", default="val")
    c.add_argument("--weights", default="models/yolo_damage_v1.pt")
    c.add_argument("--method", default="yolo",
                   choices=["classical", "yolo", "permissive"])
    c.add_argument("--max-false-alarms", type=float, default=0.05,
                   help="fraction of clean frames allowed to raise an alert")
    c.add_argument("--out", default=None, help="write the sweep to this JSON file")
    c.set_defaults(func=cmd_calibrate)

    g = sub.add_parser("gate", help="release gate — exits non-zero if the model may not ship")
    g.add_argument("--data", required=True, help="dataset directory from `onboard`")
    g.add_argument("--split", default="test")
    g.add_argument("--weights", default="models/yolo_damage_v1.pt")
    g.add_argument("--method", default="yolo",
                   choices=["classical", "yolo", "permissive"],
                   help="'permissive' uses the torchvision (BSD-3) detector — "
                        "no Ultralytics/AGPL in the inference path")
    g.add_argument("--conf", type=float, default=0.25)
    g.add_argument("--iou", type=float, default=0.5)
    g.add_argument("--max-false-alarms", type=float, default=0.05)
    g.add_argument("--min-recall", type=float, default=0.80)
    g.add_argument("--min-precision", type=float, default=None)
    g.add_argument("--min-clean-frames", type=int, default=50)
    g.add_argument("--min-damaged-frames", type=int, default=50)
    g.add_argument("--operating-point", default=None,
                   help="YAML/JSON file holding the acceptance contract "
                        "(overrides the flags above; keep it in version control)")
    g.add_argument("--out", default=None, help="write the verdict to this JSON file")
    g.set_defaults(func=cmd_gate)

    # Registered so they appear in --help; main() routes them before argparse
    # sees their arguments, so everything after the name reaches the script.
    for name, (script, help_text) in WRAPPED.items():
        sp = sub.add_parser(name, help=help_text, add_help=False)
        sp.set_defaults(func=lambda a, s=script: _delegate(s, a.rest))
        sp.add_argument("rest", nargs=argparse.REMAINDER)

    v = sub.add_parser("version", help="print the version")
    v.set_defaults(func=cmd_version)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Wrapped commands are routed first so their flags are never parsed here.
    if argv and argv[0] in WRAPPED:
        return _delegate(WRAPPED[argv[0]][0], argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) and not getattr(args, "command", None):
        return cmd_version(args)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

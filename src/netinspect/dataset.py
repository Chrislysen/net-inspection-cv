"""Bring your own footage: ingest, audit and split a real net-inspection dataset.

This is the door a farm or supplier walks through. They arrive with a folder of
images and labels in whatever their annotation tool exports, and they need to
leave with a dataset that can be trained on and, crucially, *evaluated honestly*.

Formats are detected rather than declared — YOLO ``.txt``, COCO ``.json``, Pascal
VOC ``.xml``, or images with no labels at all (which is still useful: the
label-free anomaly path and the OOD gate both train on normal footage).

Why the audit exists
--------------------
The failure that ruins a net-inspection model is never a crash. It is a number
that looks good and means nothing, and there are two reliable ways to produce
one:

**Split leakage.** Underwater inspection footage is video. Consecutive frames are
nearly identical, so a random image-level split puts frame 100 in train and frame
101 in test, and the model is graded on pictures it has already memorised. F1
comes back at 0.99 and the model fails on the next pen. Splitting is therefore
**grouped by clip** by default, and the report states how many groups landed in
each split — because a "grouped" split over two clips is not a split at all.

**Near-duplicate frames across splits.** Grouping by clip does not save you if
the same scene was exported twice under different names, which happens whenever
someone re-runs an extraction. Every image is perceptually hashed and
cross-split collisions are reported.

Neither check is exotic. Both are omitted often enough that this module treats
them as the point rather than a nicety.

Nothing here silently repairs data. Problems are reported with the file that
caused them, because a label set that needs fixing is the annotator's problem to
fix, not something to paper over.
"""
from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .utils import get_logger

LOGGER = get_logger()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Frames extracted from one clip share a prefix and differ only by an index —
# "<clip>_000123.jpg". Everything up to that trailing counter is the group, so
# an entire clip moves between splits together.
CLIP_PATTERN = re.compile(r"^(?P<group>.+?)[_-]?\d{3,}$")

# A box smaller than this fraction of the image is almost always a mis-click or
# a stray keystroke rather than a defect somebody meant to draw.
MIN_BOX_FRACTION = 1e-5


@dataclass
class Box:
    """One annotation in normalised xyxy, so every format converges here."""
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Sample:
    image: Path
    boxes: list[Box] = field(default_factory=list)
    group: str = ""
    width: int = 0
    height: int = 0

    @property
    def is_negative(self) -> bool:
        """A frame of clean net. Not a missing label — an informative example."""
        return not self.boxes


# --------------------------------------------------------------------------- #
# Format detection and loading
# --------------------------------------------------------------------------- #
def group_of(path: Path, pattern: re.Pattern = CLIP_PATTERN) -> str:
    """Which clip a frame came from, inferred from its name."""
    m = pattern.match(path.stem)
    return m.group("group") if m else path.stem


def detect_format(root: Path) -> str:
    """Guess the annotation format from what is actually on disk."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No such dataset directory: {root}")
    if any(root.rglob("*.json")):
        for j in root.rglob("*.json"):
            try:
                head = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(head, dict) and "annotations" in head and "images" in head:
                return "coco"
    if any(root.rglob("*.xml")):
        return "voc"
    if any(root.rglob("*.txt")):
        return "yolo"
    return "images"


def _images_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.width, im.height


def _load_yolo(root: Path, images: Sequence[Path]) -> list[Sample]:
    """YOLO: one .txt beside (or in a parallel labels/ tree) each image."""
    out = []
    for img in images:
        candidates = [img.with_suffix(".txt")]
        # The conventional layout puts labels in a sibling tree.
        parts = list(img.parts)
        if "images" in parts:
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            candidates.append(Path(*parts).with_suffix(".txt"))
        label = next((c for c in candidates if c.exists()), None)
        w, h = _image_size(img)
        boxes = []
        if label is not None:
            for line in label.read_text(encoding="utf-8").splitlines():
                bits = line.split()
                if len(bits) < 5:
                    continue
                cls, cx, cy, bw, bh = (float(b) for b in bits[:5])
                boxes.append(Box(int(cls), cx - bw / 2, cy - bh / 2,
                                 cx + bw / 2, cy + bh / 2))
        out.append(Sample(image=img, boxes=boxes, group=group_of(img), width=w, height=h))
    return out


def _load_coco(root: Path) -> list[Sample]:
    ann_file = next((j for j in sorted(root.rglob("*.json"))
                     if _is_coco(j)), None)
    if ann_file is None:
        raise ValueError(f"No COCO annotation file found under {root}")
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    by_id = {im["id"]: im for im in data.get("images", [])}
    anns: dict[int, list[Box]] = defaultdict(list)
    for a in data.get("annotations", []):
        im = by_id.get(a["image_id"])
        if not im:
            continue
        w, h = float(im["width"]), float(im["height"])
        x, y, bw, bh = a["bbox"]
        anns[a["image_id"]].append(
            Box(int(a.get("category_id", 0)), x / w, y / h, (x + bw) / w, (y + bh) / h))

    lookup = {p.name: p for p in _images_under(root)}
    out = []
    for im in data.get("images", []):
        path = lookup.get(Path(im["file_name"]).name)
        if path is None:
            continue
        out.append(Sample(image=path, boxes=anns.get(im["id"], []),
                          group=group_of(path),
                          width=int(im["width"]), height=int(im["height"])))
    return out


def _is_coco(p: Path) -> bool:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(d, dict) and "annotations" in d and "images" in d


def _load_voc(root: Path, images: Sequence[Path]) -> list[Sample]:
    classes: dict[str, int] = {}
    out = []
    for img in images:
        xml = img.with_suffix(".xml")
        w, h = _image_size(img)
        boxes = []
        if xml.exists():
            tree = ET.parse(xml)
            for obj in tree.getroot().findall("object"):
                name = (obj.findtext("name") or "damage").strip()
                cid = classes.setdefault(name, len(classes))
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                x1 = float(bb.findtext("xmin", "0")) / w
                y1 = float(bb.findtext("ymin", "0")) / h
                x2 = float(bb.findtext("xmax", "0")) / w
                y2 = float(bb.findtext("ymax", "0")) / h
                boxes.append(Box(cid, x1, y1, x2, y2))
        out.append(Sample(image=img, boxes=boxes, group=group_of(img), width=w, height=h))
    return out


def load_dataset(root: str | Path, fmt: str | None = None) -> list[Sample]:
    """Load a dataset in whatever format it happens to be in."""
    root = Path(root)
    fmt = fmt or detect_format(root)
    images = _images_under(root)
    if not images:
        raise ValueError(f"No images found under {root}")
    LOGGER.info("Loading %d images from %s as %s", len(images), root, fmt)
    if fmt == "coco":
        return _load_coco(root)
    if fmt == "voc":
        return _load_voc(root, images)
    if fmt == "yolo":
        return _load_yolo(root, images)
    # Label-free ingestion still has to size the frames. Returning 0x0 here made
    # audit() report every image as "could not be sized" at error severity, so
    # `netinspect onboard` refused a plain folder of footage — the one path a
    # company brings its own data through. Reading the header costs the same
    # PIL.open the other loaders already do.
    out = []
    for p in images:
        try:
            w, h = _image_size(p)
        except Exception:                 # genuinely unreadable: let audit say so
            w = h = 0
        out.append(Sample(image=p, group=group_of(p), width=w, height=h))
    return out


# --------------------------------------------------------------------------- #
# Perceptual hashing — for cross-split duplicate detection
# --------------------------------------------------------------------------- #
def dhash(path: str | Path, size: int = 8) -> int:
    """A 64-bit difference hash: near-identical frames collide, resizes do not matter.

    Cheap on purpose. The job is not image retrieval, it is catching the same
    scene appearing on both sides of a train/test split, and a gradient hash does
    that with no extra dependency.
    """
    from PIL import Image
    with Image.open(path) as im:
        small = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        px = small.tobytes()          # row-major grayscale; getdata() is deprecated
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] < px[base + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    kind: str
    detail: str
    files: list[str] = field(default_factory=list)
    severity: str = "warning"          # "error" blocks, "warning" informs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit(samples: Sequence[Sample], max_examples: int = 8) -> list[Issue]:
    """Everything wrong with this dataset, with the files that caused it.

    Errors are things that will corrupt training or evaluation. Warnings are
    things a person should look at before believing a number.
    """
    issues: list[Issue] = []

    def add(kind, detail, files, severity="warning"):
        issues.append(Issue(kind, detail, [str(f) for f in files[:max_examples]], severity))

    if not samples:
        add("empty", "No images were loaded.", [], "error")
        return issues

    bad_geom, tiny, out_of_range, unreadable = [], [], [], []
    for s in samples:
        if s.width <= 0 or s.height <= 0:
            unreadable.append(s.image)
        for b in s.boxes:
            if b.x2 <= b.x1 or b.y2 <= b.y1:
                bad_geom.append(s.image)
            elif b.area < MIN_BOX_FRACTION:
                tiny.append(s.image)
            if min(b.x1, b.y1) < -1e-6 or max(b.x2, b.y2) > 1 + 1e-6:
                out_of_range.append(s.image)

    if unreadable:
        add("unreadable", f"{len(unreadable)} image(s) could not be sized.", unreadable, "error")
    if bad_geom:
        add("degenerate_box",
            f"{len(bad_geom)} box(es) have zero or negative area — the annotation "
            "tool exported a click, not a region.", bad_geom, "error")
    if out_of_range:
        add("box_out_of_bounds",
            f"{len(out_of_range)} box(es) fall outside the image. Usually a "
            "pixel/normalised unit mix-up.", out_of_range, "error")
    if tiny:
        add("suspiciously_small",
            f"{len(tiny)} box(es) are under {MIN_BOX_FRACTION:.0e} of the frame.", tiny)

    labelled = [s for s in samples if s.boxes]
    negatives = [s for s in samples if not s.boxes]
    if not labelled:
        add("no_labels",
            "No annotations at all. The supervised detector cannot be trained; "
            "the label-free anomaly path and the OOD gate still can.", [], "warning")

    groups = {s.group for s in samples}
    if len(groups) < 3:
        add("too_few_groups",
            f"Only {len(groups)} distinct clip(s)/group(s). A grouped split needs "
            "several independent clips; with this few, held-out numbers describe "
            "one scene rather than the site.", sorted(groups), "warning")

    counts = Counter(b.cls for s in samples for b in s.boxes)
    if len(counts) > 1:
        rarest = min(counts.values())
        if rarest < 20:
            add("class_imbalance",
                f"Rarest class has only {rarest} instance(s); per-class metrics "
                "will be noise.", [])

    if labelled:
        # Guarded on `labelled`, not on `negatives`: the case worth shouting
        # about is zero clean frames, and requiring negatives to exist made the
        # check silently unreachable in exactly that case.
        ratio = len(negatives) / len(samples)
        if ratio < 0.1:
            add("few_negatives",
                f"Only {ratio:.0%} of frames are clean net. False-alarm rate is "
                "the number that decides whether operators trust this, and it "
                "cannot be measured without clean frames.", [])

    return issues


def find_cross_split_duplicates(splits: dict[str, list[Sample]],
                                threshold: int = 4) -> list[tuple[str, str]]:
    """Near-identical frames appearing in two different splits.

    This is the quiet one. Grouping by clip stops consecutive frames straddling
    a split, but not the same footage exported twice under different names, and
    the resulting score is meaningless in a way nothing else will reveal.
    """
    hashed: list[tuple[str, Path, int]] = []
    for name, samples in splits.items():
        for s in samples:
            try:
                hashed.append((name, s.image, dhash(s.image)))
            except Exception as exc:
                LOGGER.debug("hash failed for %s: %s", s.image, exc)
    dupes = []
    for i in range(len(hashed)):
        ni, pi, hi = hashed[i]
        for j in range(i + 1, len(hashed)):
            nj, pj, hj = hashed[j]
            if ni != nj and hamming(hi, hj) <= threshold:
                dupes.append((str(pi), str(pj)))
    return dupes


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_samples(samples: Sequence[Sample], ratios=(0.7, 0.15, 0.15),
                  seed: int = 0, group: bool = True) -> dict[str, list[Sample]]:
    """Split into train/val/test, keeping whole clips together by default.

    Grouped is the default because the ungrouped alternative is the single most
    common way an inspection model is accidentally graded on its own training
    frames. ``group=False`` exists for datasets of genuinely independent stills,
    and the health report always records which was used.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {sum(ratios)}")
    rng = random.Random(seed)

    if group:
        by_group: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_group[s.group].append(s)
        units: list[list[Sample]] = list(by_group.values())
    else:
        units = [[s] for s in samples]
    rng.shuffle(units)

    total = sum(len(u) for u in units)
    want = {"train": ratios[0] * total, "val": ratios[1] * total, "test": ratios[2] * total}
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    ordered = sorted(units, key=len, reverse=True)

    # Seed every requested split with one unit first. Pure greedy-by-deficit
    # starves the small splits when there are only a handful of large clips: four
    # 40-frame clips at 70/15/15 put three in train, one in val and left the test
    # split empty. An empty test split is not a split.
    wanted = [k for k in ("train", "val", "test") if want[k] > 0]
    for name in wanted:
        if ordered:
            out[name].extend(ordered.pop(len(ordered) // 2 if name != "train" else 0))

    for unit in ordered:
        name = max(out, key=lambda k: want[k] - len(out[k]))
        out[name].extend(unit)
    return out


# --------------------------------------------------------------------------- #
# Writing a trainable dataset
# --------------------------------------------------------------------------- #
def write_yolo(splits: dict[str, list[Sample]], out_dir: str | Path,
               class_names: Sequence[str] = ("damage",),
               copy_images: bool = True) -> Path:
    """Write the split out in YOLO layout and return the dataset.yaml path."""
    import shutil

    out_dir = Path(out_dir)
    for name, samples in splits.items():
        img_dir = out_dir / "images" / name
        lbl_dir = out_dir / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            dst = img_dir / s.image.name
            if copy_images and not dst.exists():
                shutil.copy2(s.image, dst)
            lines = []
            for b in s.boxes:
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                bw, bh = b.x2 - b.x1, b.y2 - b.y1
                lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_dir / f"{s.image.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    yaml_path = out_dir / "dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(
        f"# Generated by netinspect onboard — do not hand-edit.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n", encoding="utf-8")
    return yaml_path


def health_report(samples: Sequence[Sample], splits: dict[str, list[Sample]],
                  issues: Sequence[Issue], duplicates: Sequence[tuple[str, str]],
                  grouped: bool) -> dict[str, Any]:
    """The document that decides whether any later number is worth believing."""
    per_split = {k: {"images": len(v),
                     "boxes": sum(len(s.boxes) for s in v),
                     "clean_frames": sum(1 for s in v if s.is_negative),
                     "groups": len({s.group for s in v})}
                 for k, v in splits.items()}
    issues = list(issues)
    empty = [k for k, v in per_split.items() if not v["images"]]
    if empty:
        issues.append(Issue(
            "empty_split",
            f"Split(s) {', '.join(empty)} received no frames — there are too few "
            "independent clips to divide. Held-out evaluation is impossible; "
            "collect footage from more passes before trusting any score.",
            [], "error"))
    errors = [i for i in issues if i.severity == "error"]
    return {
        "images": len(samples),
        "boxes": sum(len(s.boxes) for s in samples),
        "clean_frames": sum(1 for s in samples if s.is_negative),
        "groups": len({s.group for s in samples}),
        "class_counts": dict(Counter(b.cls for s in samples for b in s.boxes)),
        "split_strategy": "grouped by clip" if grouped else "random per image",
        "splits": per_split,
        "cross_split_duplicates": len(duplicates),
        "duplicate_examples": [list(d) for d in duplicates[:10]],
        "issues": [i.to_dict() for i in issues],
        "blocking_errors": len(errors),
        "usable": not errors,
        "note": (
            "A grouped split is the default because inspection footage is video: "
            "a random image-level split grades the model on frames adjacent to "
            "its own training data. Cross-split duplicates are reported "
            "separately because grouping does not catch footage exported twice."),
    }


def summarise(report: dict[str, Any]) -> str:
    """A short human summary — the thing an engineer reads before training."""
    lines = [
        f"{report['images']} images · {report['boxes']} boxes · "
        f"{report['clean_frames']} clean frames · {report['groups']} clip(s)",
        f"split: {report['split_strategy']}",
    ]
    for name, s in report["splits"].items():
        lines.append(f"  {name:5s} {s['images']:5d} images  {s['boxes']:5d} boxes  "
                     f"{s['groups']:3d} clip(s)")
    if report["cross_split_duplicates"]:
        lines.append(f"  !! {report['cross_split_duplicates']} near-duplicate pair(s) "
                     "span two splits — held-out scores will be inflated")
    for i in report["issues"]:
        mark = "ERROR" if i["severity"] == "error" else "warn "
        lines.append(f"  [{mark}] {i['detail']}")
    return "\n".join(lines)


__all__ = ["Box", "Sample", "Issue", "load_dataset", "detect_format", "group_of",
           "audit", "split_samples", "write_yolo", "health_report", "summarise",
           "dhash", "hamming", "find_cross_split_duplicates",
           "IMAGE_SUFFIXES", "CLIP_PATTERN"]

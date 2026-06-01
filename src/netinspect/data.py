"""Dataset discovery, YOLO label parsing, and dataset summaries.

Label formats supported (both are the standard Ultralytics/YOLO text format,
one ``.txt`` per image with the same stem):

* **Detection**: ``class_id xc yc w h``  (all normalised to [0, 1])
* **Segmentation**: ``class_id x1 y1 x2 y2 ... xn yn``  (normalised polygon)

A missing label file is treated as "no annotations for this image" (which is
valid for unlabelled data), while a malformed label file is reported as an
error so data problems surface early instead of silently corrupting metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .utils import BBox, IMAGE_EXTENSIONS, image_size, list_images

# Default class map for the prototype. Real projects should define this with
# domain experts; see configs/baseline.yaml.
DEFAULT_CLASSES = ["damage", "hole", "tear", "abnormal_region"]


@dataclass
class Polygon:
    """A segmentation polygon in absolute pixel coordinates."""
    points: list[tuple[float, float]]
    class_id: int = 0
    class_name: str = "damage"
    score: float = 1.0


@dataclass
class Sample:
    """One image plus its (optional) annotations."""
    image_path: Path
    width: int
    height: int
    boxes: list[BBox] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    label_path: Path | None = None
    label_kind: str = "none"  # "none" | "detection" | "segmentation"
    label_errors: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def has_labels(self) -> bool:
        return bool(self.boxes or self.polygons)


def _class_name(classes: list[str], idx: int) -> str:
    return classes[idx] if 0 <= idx < len(classes) else f"class_{idx}"


def parse_yolo_label(
    label_path: Path,
    width: int,
    height: int,
    classes: list[str] | None = None,
) -> tuple[list[BBox], list[Polygon], str, list[str]]:
    """Parse a YOLO label file, auto-detecting detection vs segmentation.

    Returns ``(boxes, polygons, kind, errors)``. Detection lines have exactly
    5 fields; segmentation lines have an odd number >= 7 (class + >=3 xy pairs).
    """
    classes = classes or DEFAULT_CLASSES
    boxes: list[BBox] = []
    polygons: list[Polygon] = []
    errors: list[str] = []
    kinds: set[str] = set()

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, polygons, "none", errors

    for lineno, raw in enumerate(text.splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{label_path.name}:{lineno}: non-numeric token")
            continue

        if len(values) == 4:  # detection: xc yc w h (normalised)
            xc, yc, w, h = values
            if not all(0.0 <= v <= 1.0 for v in values):
                errors.append(f"{label_path.name}:{lineno}: box not normalised to [0,1]")
            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height
            boxes.append(BBox(x1, y1, x2, y2, class_id,
                              _class_name(classes, class_id), 1.0))
            kinds.add("detection")
        elif len(values) >= 6 and len(values) % 2 == 0:  # polygon: x1 y1 ... xn yn
            pts = [(values[i] * width, values[i + 1] * height)
                   for i in range(0, len(values), 2)]
            polygons.append(Polygon(pts, class_id,
                                    _class_name(classes, class_id), 1.0))
            kinds.add("segmentation")
        else:
            errors.append(
                f"{label_path.name}:{lineno}: expected 4 (box) or even>=6 "
                f"(polygon) coords, got {len(values)}"
            )

    if kinds == {"detection"}:
        kind = "detection"
    elif kinds == {"segmentation"}:
        kind = "segmentation"
    elif len(kinds) > 1:
        kind = "mixed"
        errors.append(f"{label_path.name}: mixes detection and segmentation lines")
    else:
        kind = "none"
    return boxes, polygons, kind, errors


def find_label_path(image_path: Path, labels_dir: Path | None) -> Path | None:
    """Locate the YOLO ``.txt`` label for an image, if one exists.

    Mirrors the Ultralytics convention: either a sibling ``labels/`` directory
    next to ``images/``, or an explicit labels directory.
    """
    candidates: list[Path] = []
    if labels_dir is not None:
        candidates.append(labels_dir / f"{image_path.stem}.txt")
    # sibling images/ -> labels/ convention
    if image_path.parent.name == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.with_suffix(".txt"))
    for c in candidates:
        if c.exists():
            return c
    return None


def load_dataset(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    classes: list[str] | None = None,
) -> list[Sample]:
    """Scan an images directory and attach labels where present."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else None
    classes = classes or DEFAULT_CLASSES

    samples: list[Sample] = []
    for img_path in list_images(images_dir):
        try:
            w, h = image_size(img_path)
        except Exception as exc:  # unreadable / corrupt image
            samples.append(Sample(img_path, 0, 0, label_errors=[f"unreadable image: {exc}"]))
            continue

        label_path = find_label_path(img_path, labels_dir)
        if label_path is None:
            samples.append(Sample(img_path, w, h))
            continue

        boxes, polygons, kind, errors = parse_yolo_label(label_path, w, h, classes)
        samples.append(Sample(img_path, w, h, boxes, polygons,
                              label_path, kind, errors))
    return samples


def summarize_dataset(samples: list[Sample], classes: list[str] | None = None) -> dict:
    """Compute a structured summary: counts, sizes, class distribution, errors."""
    classes = classes or DEFAULT_CLASSES
    sizes = [(s.width, s.height) for s in samples if s.width and s.height]
    labelled = [s for s in samples if s.has_labels]
    class_counts: dict[str, int] = {c: 0 for c in classes}
    box_count = poly_count = 0
    errors: list[str] = []
    kinds: dict[str, int] = {}

    for s in samples:
        kinds[s.label_kind] = kinds.get(s.label_kind, 0) + 1
        for b in s.boxes:
            class_counts[_class_name(classes, b.class_id)] = \
                class_counts.get(_class_name(classes, b.class_id), 0) + 1
            box_count += 1
        for p in s.polygons:
            class_counts[_class_name(classes, p.class_id)] = \
                class_counts.get(_class_name(classes, p.class_id), 0) + 1
            poly_count += 1
        errors.extend(f"{s.image_path.name}: {e}" for e in s.label_errors)

    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    return {
        "num_images": len(samples),
        "num_labelled_images": len(labelled),
        "num_unlabelled_images": len(samples) - len(labelled),
        "num_boxes": box_count,
        "num_polygons": poly_count,
        "label_kinds": kinds,
        "class_distribution": class_counts,
        "image_size": {
            "min": [min(widths), min(heights)] if sizes else None,
            "max": [max(widths), max(heights)] if sizes else None,
            "unique_sizes": sorted({f"{w}x{h}" for w, h in sizes}),
        },
        "num_label_errors": len(errors),
        "label_errors": errors[:50],  # cap to keep summaries readable
    }


def format_summary(summary: dict) -> str:
    """Render a dataset summary as a human-readable text block."""
    lines = [
        "Dataset summary",
        "===============",
        f"Images:            {summary['num_images']}",
        f"  with labels:     {summary['num_labelled_images']}",
        f"  without labels:  {summary['num_unlabelled_images']}",
        f"Bounding boxes:    {summary['num_boxes']}",
        f"Polygons (masks):  {summary['num_polygons']}",
        f"Label kinds:       {summary['label_kinds']}",
        "Class distribution:",
    ]
    for cls, count in summary["class_distribution"].items():
        lines.append(f"  {cls:<18} {count}")
    sz = summary["image_size"]
    if sz["min"]:
        lines.append(f"Image size (WxH):  min={sz['min']}  max={sz['max']}")
        lines.append(f"  unique sizes:    {sz['unique_sizes']}")
    lines.append(f"Label errors:      {summary['num_label_errors']}")
    for err in summary["label_errors"]:
        lines.append(f"  ! {err}")
    return "\n".join(lines)

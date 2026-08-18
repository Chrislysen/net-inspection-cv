"""COCO-format ingestion: convert COCO detection/segmentation JSON to YOLO labels.

Why this exists
---------------
Real labelled data arrives in COCO far more often than YOLO. When an operator (or an
annotation vendor) provides labelled net-damage frames, they will most likely be
COCO. This adapter is that drop-in slot: point it at a COCO JSON + image folder
and it writes a standard YOLO dataset the rest of the pipeline already consumes.

Honest scoping for public underwater datasets
----------------------------------------------
Public underwater sets (SeaClear, TrashCan, Trash-ICRA19) are marine **debris**
(plastic, animals, plants), **not** net damage. They are useful as *related-
domain* data — transfer-learning pretraining, a real-image smoke test of this
adapter, and negative/background variety — but training on them does **not**
produce a net-damage detector, and they should not be reported as a "damage
proxy". Always verify each dataset's licence before downloading/redistributing;
do not assume it from secondary summaries.

No pycocotools dependency — plain JSON parsing keeps this lightweight.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .utils import ensure_dir, get_logger

LOGGER = get_logger()


@dataclass
class CocoConvertResult:
    out_dir: Path
    num_images: int = 0
    num_labels: int = 0
    num_instances: int = 0
    class_names: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _normalise_bbox(bbox: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] (pixels, top-left) -> YOLO [xc, yc, w, h] (normalised)."""
    x, y, bw, bh = bbox
    return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)


def _polygon_to_norm(seg: list[float], w: int, h: int) -> list[float]:
    """COCO flat polygon [x1,y1,...] (pixels) -> normalised [x1,y1,...]."""
    out = []
    for i in range(0, len(seg) - 1, 2):
        out.append(seg[i] / w)
        out.append(seg[i + 1] / h)
    return out


def convert_coco_to_yolo(
    coco_json: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    segmentation: bool = False,
    class_map: dict[int, int] | None = None,
    single_class: bool = False,
    copy_images: bool = True,
) -> CocoConvertResult:
    """Convert a COCO annotation file to a YOLO dataset under ``out_dir``.

    Parameters
    ----------
    segmentation : bool
        Write polygon (seg) labels instead of boxes when ``segmentation`` masks
        are present in the COCO file.
    class_map : dict, optional
        Remap COCO ``category_id`` -> contiguous YOLO class index. If omitted,
        categories are mapped to 0..N-1 in sorted id order.
    single_class : bool
        Collapse every annotation to class 0 (e.g. a single "object"/"damage"
        class). Overrides ``class_map``.
    """
    coco_json, images_dir, out_dir = Path(coco_json), Path(images_dir), Path(out_dir)
    data = json.loads(coco_json.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in data.get("images", [])}
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    if single_class:
        class_map = {c["id"]: 0 for c in cats}
        class_names = ["object"]
    else:
        if class_map is None:
            class_map = {c["id"]: i for i, c in enumerate(cats)}
        idx_to_name = {class_map[c["id"]]: c["name"] for c in cats if c["id"] in class_map}
        class_names = [idx_to_name.get(i, f"class_{i}") for i in range(max(idx_to_name) + 1)] \
            if idx_to_name else []

    anns_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out = ensure_dir(out_dir / "images")
    lbl_out = ensure_dir(out_dir / "labels")
    res = CocoConvertResult(out_dir=out_dir, class_names=class_names)

    for img_id, img in images.items():
        w, h = img.get("width"), img.get("height")
        fname = img["file_name"]
        src = images_dir / fname
        if not src.exists() or not w or not h:
            res.skipped.append(fname)
            continue
        if copy_images:
            shutil.copy2(src, img_out / Path(fname).name)
        res.num_images += 1

        lines = []
        for ann in anns_by_image.get(img_id, []):
            if ann.get("iscrowd"):
                continue
            cid = class_map.get(ann["category_id"])
            if cid is None:
                continue
            if segmentation and ann.get("segmentation"):
                seg = ann["segmentation"]
                polys = seg if isinstance(seg, list) and seg and isinstance(seg[0], list) else [seg]
                for poly in polys:
                    if isinstance(poly, list) and len(poly) >= 6:
                        coords = " ".join(f"{v:.6f}" for v in _polygon_to_norm(poly, w, h))
                        lines.append(f"{cid} {coords}")
                        res.num_instances += 1
            elif ann.get("bbox"):
                xc, yc, bw, bh = _normalise_bbox(ann["bbox"], w, h)
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                res.num_instances += 1

        if lines:
            (lbl_out / f"{Path(fname).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            res.num_labels += 1

    # Ultralytics dataset YAML.
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names)) or "  0: object"
    (out_dir / "dataset.yaml").write_text(
        f"# Converted from COCO by netinspect.coco. Verify dataset licence before use.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images\nval: images\nnames:\n{names_yaml}\n", encoding="utf-8")

    LOGGER.info("COCO->YOLO: %d images, %d labelled, %d instances, %d classes (%d skipped)",
                res.num_images, res.num_labels, res.num_instances,
                len(class_names), len(res.skipped))
    return res

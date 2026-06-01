"""Shared utilities: logging, IO, geometry, and optional-dependency handling.

This module intentionally has *no* hard dependency on OpenCV. Heavy/optional
libraries (cv2, ultralytics, skimage) are imported lazily by the modules that
need them so that the package can still be imported, and the data/metric code
still runs, in a minimal environment.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def get_logger(name: str = "netinspect", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


LOGGER = get_logger()


def optional_import(module_name: str):
    """Import a module if available, otherwise return None.

    Used to keep optional dependencies (cv2, ultralytics, skimage) soft so the
    rest of the toolkit degrades gracefully instead of crashing on import.
    """
    try:
        return import_module(module_name)
    except Exception:  # ImportError or a broken install
        return None


def require(module_name: str, hint: str | None = None):
    """Import a module or raise a clear, actionable error."""
    mod = optional_import(module_name)
    if mod is None:
        msg = f"Required dependency '{module_name}' is not installed."
        if hint:
            msg += f" {hint}"
        raise RuntimeError(msg)
    return mod


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def list_images(directory: str | Path) -> list[Path]:
    """Return sorted image paths in a directory (non-recursive)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates.

    Attributes
    ----------
    x1, y1, x2, y2 : float
        Top-left and bottom-right corners (x2 >= x1, y2 >= y1).
    class_id : int
        Integer class index.
    class_name : str
        Human-readable class name.
    score : float
        Confidence in [0, 1]. Ground-truth boxes use 1.0.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    class_name: str = "damage"
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a, b) -> float:
    """IoU of two boolean/0-1 numpy masks of the same shape."""
    import numpy as np
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Image IO
#
# We standardise on RGB uint8 numpy arrays everywhere in the codebase. cv2 is
# used when available (faster, fewer deps for video), otherwise we fall back to
# Pillow so the core pipeline keeps working in a minimal environment.
# --------------------------------------------------------------------------- #
def image_size(path: str | Path) -> tuple[int, int]:
    """Return (width, height) without decoding the full image when possible."""
    pil = optional_import("PIL.Image")
    if pil is not None:
        with pil.open(path) as im:  # type: ignore[union-attr]
            return int(im.width), int(im.height)
    img = read_image(path)
    return img.shape[1], img.shape[0]


def read_image(path: str | Path):
    """Read an image as an RGB uint8 numpy array (H, W, 3)."""
    import numpy as np
    cv2 = optional_import("cv2")
    if cv2 is not None:
        # imread does not handle non-ASCII paths on Windows; read bytes first.
        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = require("PIL.Image", hint="Install opencv-python-headless or Pillow.")
    with pil.open(path) as im:
        return np.asarray(im.convert("RGB"))


def write_image(path: str | Path, image) -> None:
    """Write an RGB uint8 numpy array to disk."""
    import numpy as np
    path = Path(path)
    ensure_dir(path.parent)
    image = np.asarray(image)
    cv2 = optional_import("cv2")
    if cv2 is not None:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(path.suffix or ".png", bgr)
        if not ok:
            raise ValueError(f"Could not encode image: {path}")
        buf.tofile(str(path))  # unicode-safe write on Windows
        return
    pil = require("PIL.Image", hint="Install opencv-python-headless or Pillow.")
    pil.fromarray(image).save(path)


# --------------------------------------------------------------------------- #
# Prediction IO
#
# A single JSON schema is shared by the classical baseline, the YOLO inference
# script, and the evaluator:
#
#   {"meta": {...}, "images": {"<filename>": [ {bbox, score, class_id, class_name}, ... ]}}
# --------------------------------------------------------------------------- #
def predictions_to_dict(preds_by_image: dict[str, list["BBox"]],
                        meta: dict | None = None) -> dict:
    return {
        "meta": meta or {},
        "images": {
            name: [
                {"bbox": [round(v, 2) for v in b.to_list()],
                 "score": round(b.score, 4),
                 "class_id": b.class_id, "class_name": b.class_name}
                for b in boxes
            ]
            for name, boxes in preds_by_image.items()
        },
    }


def save_predictions(preds_by_image: dict[str, list["BBox"]], path: str | Path,
                     meta: dict | None = None) -> None:
    write_json(predictions_to_dict(preds_by_image, meta), path)


def load_predictions(path: str | Path) -> dict[str, list["BBox"]]:
    raw = read_json(path)
    images = raw.get("images", raw)  # tolerate a bare {filename: [...]} mapping
    out: dict[str, list[BBox]] = {}
    for name, dets in images.items():
        boxes = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append(BBox(x1, y1, x2, y2,
                              int(d.get("class_id", 0)),
                              d.get("class_name", "damage"),
                              float(d.get("score", 1.0))))
        out[name] = boxes
    return out

"""Visualisation: overlays, side-by-side comparisons, and a results gallery.

Domain experts usually trust what they can *see* before they trust a metric, so
these helpers exist to make predictions and failure cases easy to eyeball.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .utils import BBox, ensure_dir, optional_import, write_image

# Distinct, colour-blind-friendly-ish RGB colours.
_GT_COLOR = (0, 200, 0)        # green = ground truth
_PRED_COLOR = (255, 80, 0)     # orange = prediction
_FP_COLOR = (220, 0, 0)        # red = false positive
_FN_COLOR = (0, 120, 255)      # blue = false negative (missed)


def _draw_box(img: np.ndarray, box: BBox, color, label: str | None = None,
              thickness: int = 2) -> None:
    cv2 = optional_import("cv2")
    x1, y1, x2, y2 = (int(round(v)) for v in box.to_list())
    if cv2 is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:  # NumPy fallback: draw a simple rectangle outline.
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        for t in range(thickness):
            if y1 + t < h:
                img[y1 + t, x1:x2] = color
            if y2 - t >= 0:
                img[y2 - t, x1:x2] = color
            if x1 + t < w:
                img[y1:y2, x1 + t] = color
            if x2 - t >= 0:
                img[y1:y2, x2 - t] = color


def overlay_boxes(image_rgb: np.ndarray, preds: Sequence[BBox] = (),
                  gts: Sequence[BBox] = (), show_scores: bool = True) -> np.ndarray:
    """Return a copy of the image with prediction (orange) and GT (green) boxes."""
    out = image_rgb.copy()
    for g in gts:
        _draw_box(out, g, _GT_COLOR, "GT")
    for p in preds:
        label = f"{p.class_name} {p.score:.2f}" if show_scores else p.class_name
        _draw_box(out, p, _PRED_COLOR, label)
    return out


def overlay_match(image_rgb: np.ndarray, preds: Sequence[BBox], gts: Sequence[BBox],
                  fp_idx: Sequence[int], fn_idx: Sequence[int]) -> np.ndarray:
    """Colour-code TP/FP predictions and FN (missed) ground truth boxes."""
    out = image_rgb.copy()
    fp_set, fn_set = set(fp_idx), set(fn_idx)
    for gi, g in enumerate(gts):
        if gi in fn_set:
            _draw_box(out, g, _FN_COLOR, "MISS")
    for pi, p in enumerate(preds):
        color = _FP_COLOR if pi in fp_set else _PRED_COLOR
        tag = "FP" if pi in fp_set else "TP"
        _draw_box(out, p, color, f"{tag} {p.score:.2f}")
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Concatenate two images horizontally (padded to equal height)."""
    h = max(left.shape[0], right.shape[0])
    def pad(im):
        if im.shape[0] == h:
            return im
        out = np.zeros((h, im.shape[1], 3), dtype=im.dtype)
        out[:im.shape[0]] = im
        return out
    sep = np.full((h, 4, 3), 255, dtype=left.dtype)
    return np.concatenate([pad(left), sep, pad(right)], axis=1)


def write_gallery_markdown(out_dir: str | Path, entries: list[dict],
                           title: str = "Prediction gallery") -> Path:
    """Write a simple markdown gallery referencing saved overlay images.

    ``entries`` is a list of ``{"image": <rel-path>, "caption": <str>}``.
    """
    out_dir = ensure_dir(out_dir)
    md = [f"# {title}", ""]
    if not entries:
        md.append("_No images to display._")
    for e in entries:
        md.append(f"### {e.get('caption', e['image'])}")
        md.append(f"![{e['image']}]({e['image']})")
        md.append("")
    path = out_dir / "gallery.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def save_overlay(path: str | Path, image_rgb: np.ndarray, preds: Sequence[BBox] = (),
                 gts: Sequence[BBox] = ()) -> None:
    write_image(path, overlay_boxes(image_rgb, preds, gts))

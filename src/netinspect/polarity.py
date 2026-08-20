"""Reject detections whose interior is BRIGHTER than the net around it.

Where this came from
--------------------
Three attempts to close the between-clip false-alarm gap failed: photometric
augmentation, the Jerlov water model, and hard-negative mining on the offending
clip (which made it measurably worse). Rather than try a fourth,
``scripts/analyse_false_alarms.py`` measured what the detector actually fires on,
and the answer contradicted the repo's own stated explanation.

The false alarms *are* the mooring cords — a contact sheet of the 24
highest-scoring ones shows bright rope in 22 of them. But they are **bright**,
and the damage the detector was trained on is **dark**:

    signed contrast (surround minus interior; >0 means darker than its surround)

        real damage   p10 +15.0   median +42.3   p90 +57.1   (n=284)
        false alarms  p10 -43.8   median  -5.7   p90 +23.0   (n=109)

So the model is not firing on cords because they resemble damage in brightness —
it is firing across the polarity boundary, most likely on the oriented-edge
signature that a bright rope and a dark tear share, since both interrupt an
otherwise regular mesh. That reading also explains why all three earlier fixes
failed: two changed appearance without changing edge structure, and the third
asked the model to suppress the very signature that marks real damage too.

The physical argument, which matters more than the statistics
-------------------------------------------------------------
A real hole in a net is a hole. You see through it to unlit water, so it is
darker than the mesh around it — that is why ``netinspect.synthetic`` paints
damage dark in the first place, and it is a property of the world rather than of
the generator. A rope is an object in front of the net, lit by the same lamp that
lights the mesh, so it is brighter. The two are opposite in sign for a reason.

What this is NOT
----------------
This is a **geometric prior, not a learned improvement**, and the damage it was
tuned against is synthetic. Two limits follow, and neither is hypothetical:

* Real damage is not always dark. Backscatter, a fish behind the hole, or a
  partly-occluded tear can all raise the interior. ``reports/TECHNICAL_QA.md``
  already says real holes look different from the composited ones; this filter
  assumes exactly the property that may not transfer.
* Because the synthetic generator makes damage dark **by construction**, a
  filter tuned on it will look better on this data than it can possibly be in
  the field. Measure it on real labelled damage before trusting it.

It is therefore **off by default**. Turn it on when your damage is see-through
and your false alarms are rigging, which is the case measured here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .utils import BBox, get_logger

LOGGER = get_logger()

# Measured, not reasoned. The 10th percentile of real damage is +15, and that was
# the first guess — but sweeping it (scripts/eval_polarity_filter.py) showed 15
# already costs recall while 10 does not:
#
#   min_contrast   overall false alarms   recall
#           off                   11.5%     100%
#            10                    5.4%     100%      <- free
#            15                    4.3%      96%
#            30                    0.5%      88%
#
# So the default is the largest threshold that was measured to cost nothing. Push
# it higher deliberately, with the trade in front of you, not by accident.
DEFAULT_MIN_CONTRAST = 10.0


def luminance(image_rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma. Cheap, and the choice of coefficients does not matter here."""
    img = np.asarray(image_rgb, dtype=np.float32)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


@dataclass
class BoxContrast:
    """How much darker a box's interior is than the net immediately around it."""
    interior: float
    surround: float

    @property
    def contrast(self) -> float:
        """Positive means darker than the surround, i.e. hole-like."""
        return self.surround - self.interior


def box_contrast(image_rgb: np.ndarray, box: BBox, lum: np.ndarray | None = None) -> BoxContrast | None:
    """Signed local contrast for one detection, or None if the box is degenerate.

    The reference is a ring immediately around the box rather than the whole
    frame: underwater brightness falls off with depth and range, so a global
    mean would make every detection near the lit centre look dark and every one
    at the edge look bright.
    """
    lum = luminance(image_rgb) if lum is None else lum
    h, w = lum.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    interior = float(lum[y1:y2, x1:x2].mean())

    pad_x, pad_y = max(4, (x2 - x1) // 2), max(4, (y2 - y1) // 2)
    sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    sx2, sy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    ring = lum[sy1:sy2, sx1:sx2].astype(np.float32).copy()
    ring[y1 - sy1:y2 - sy1, x1 - sx1:x2 - sx1] = np.nan
    surround = float(np.nanmean(ring)) if np.isfinite(ring).any() else interior
    return BoxContrast(interior=interior, surround=surround)


def is_hole_like(image_rgb: np.ndarray, box: BBox,
                 min_contrast: float = DEFAULT_MIN_CONTRAST,
                 lum: np.ndarray | None = None) -> bool:
    """Is this detection darker than its surroundings by at least `min_contrast`?

    A degenerate box is kept rather than dropped: this filter exists to remove a
    specific, well-characterised false alarm, and anything it cannot measure is
    not that thing. Silently discarding the unmeasurable is how a filter starts
    eating detections nobody asked it to touch.
    """
    c = box_contrast(image_rgb, box, lum=lum)
    return True if c is None else c.contrast >= min_contrast


def filter_detections(image_rgb: np.ndarray, boxes: Sequence[BBox],
                      min_contrast: float = DEFAULT_MIN_CONTRAST) -> list[BBox]:
    """Drop detections that are brighter than the net around them.

    One luminance pass for the whole frame, reused across boxes — the naive
    version recomputes it per detection and is the difference between a
    negligible post-step and a visible one on a busy frame.
    """
    if not boxes:
        return list(boxes)
    lum = luminance(image_rgb)
    return [b for b in boxes if is_hole_like(image_rgb, b, min_contrast, lum=lum)]


__all__ = ["DEFAULT_MIN_CONTRAST", "BoxContrast", "luminance", "box_contrast",
           "is_hole_like", "filter_detections"]

"""The polarity filter: keep dark detections, drop bright ones.

This encodes a claim about the world — a hole in a net shows unlit water and is
darker than the mesh, while a rope is an object in front of the net and is
brighter — so the tests are about that claim rather than about the arithmetic.
The failure that matters is a filter which quietly eats real detections, so the
"keeps what it should" direction is tested at least as hard as the other.

Origin: `scripts/analyse_false_alarms.py` measured 109 false alarms against 284
labelled damage boxes and found them cleanly separated in signed local contrast
(damage p10 +15.0; false alarms median -5.7).
"""
from __future__ import annotations

import numpy as np
import pytest

from netinspect import polarity as POL
from netinspect.utils import BBox


def _frame(bg=120, size=200):
    return np.full((size, size, 3), bg, dtype=np.uint8)


def _paint(img, x1, y1, x2, y2, value):
    img[y1:y2, x1:x2] = value
    return img


def _box(x1, y1, x2, y2, score=0.9):
    return BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                score=score, class_name="damage")


# --------------------------------------------------------------------------- #
# the sign convention — everything else depends on getting this right
# --------------------------------------------------------------------------- #
def test_a_dark_region_has_positive_contrast():
    """Positive means darker than the surround, i.e. hole-like."""
    img = _paint(_frame(bg=120), 80, 80, 120, 120, 20)
    c = POL.box_contrast(img, _box(80, 80, 120, 120))
    assert c is not None
    assert c.contrast > 0, "a dark patch must read as positive contrast"
    assert c.interior == pytest.approx(20, abs=1)
    assert c.surround == pytest.approx(120, abs=1)


def test_a_bright_region_has_negative_contrast():
    img = _paint(_frame(bg=120), 80, 80, 120, 120, 240)
    c = POL.box_contrast(img, _box(80, 80, 120, 120))
    assert c.contrast < 0, "a bright rope must read as negative contrast"


def test_contrast_is_measured_against_the_local_ring_not_the_whole_frame():
    """Underwater brightness falls off with range.

    Against a global mean, every detection near the lit centre would look dark
    and every one at the edge would look bright — the filter would then be
    measuring vignetting rather than the object.
    """
    img = _frame(bg=30)                      # a dark frame overall
    _paint(img, 60, 60, 140, 140, 200)       # locally bright neighbourhood
    _paint(img, 90, 90, 110, 110, 150)       # target: darker than its ring, brighter than frame

    c = POL.box_contrast(img, _box(90, 90, 110, 110))
    assert c.contrast > 0, (
        "the box is darker than the ring around it and must read as hole-like, "
        "even though it is brighter than the frame mean")


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def test_bright_detections_are_dropped_and_dark_ones_kept():
    img = _frame(bg=120)
    _paint(img, 20, 20, 60, 60, 15)          # hole-like
    _paint(img, 130, 130, 170, 170, 245)     # rope-like
    dark, bright = _box(20, 20, 60, 60), _box(130, 130, 170, 170)

    kept = POL.filter_detections(img, [dark, bright], min_contrast=15.0)
    assert len(kept) == 1
    assert kept[0].x1 == dark.x1, "the wrong detection survived"


def test_the_threshold_moves_the_boundary():
    """A near-neutral box should be governed by min_contrast, not hardcoded."""
    img = _paint(_frame(bg=120), 80, 80, 120, 120, 100)   # ~20 darker than surround
    box = _box(80, 80, 120, 120)
    assert POL.filter_detections(img, [box], min_contrast=5.0), "should survive a low bar"
    assert not POL.filter_detections(img, [box], min_contrast=40.0), "should fail a high bar"


def test_an_empty_detection_list_is_returned_unchanged():
    assert POL.filter_detections(_frame(), []) == []


def test_a_degenerate_box_is_kept_rather_than_silently_eaten():
    """The filter removes one characterised false alarm; it is not a general cull.

    A box too small to measure is not the thing being targeted, so dropping it
    would be the filter deleting detections nobody asked it to touch.
    """
    img = _frame()
    sliver = _box(50, 50, 51, 51)
    assert POL.box_contrast(img, sliver) is None
    assert POL.filter_detections(img, [sliver]) == [sliver]


def test_boxes_partly_outside_the_frame_do_not_crash():
    img = _frame(size=100)
    for b in (_box(-40, -40, 30, 30), _box(70, 70, 200, 200), _box(-10, 40, 110, 60)):
        POL.filter_detections(img, [b])       # must not raise


# --------------------------------------------------------------------------- #
# the property the whole thing rests on
# --------------------------------------------------------------------------- #
def test_the_filter_separates_the_two_populations_it_was_built_from():
    """Reproduces the measured separation in miniature.

    Damage p10 was +15.0 and false alarms median -5.7, so a threshold of 15
    should keep hole-like boxes and drop rope-like ones. If this ever fails, the
    default threshold and the distribution it came from have drifted apart.
    """
    rng = np.random.default_rng(0)
    kept_dark = dropped_bright = 0
    for _ in range(30):
        img = _frame(bg=int(rng.integers(90, 160)))
        x, y = int(rng.integers(20, 140)), int(rng.integers(20, 140))
        dark = _paint(img.copy(), x, y, x + 30, y + 30, int(rng.integers(5, 60)))
        bright = _paint(img.copy(), x, y, x + 30, y + 30, int(rng.integers(200, 255)))
        b = _box(x, y, x + 30, y + 30)
        kept_dark += bool(POL.filter_detections(dark, [b], POL.DEFAULT_MIN_CONTRAST))
        dropped_bright += not POL.filter_detections(bright, [b], POL.DEFAULT_MIN_CONTRAST)

    assert kept_dark == 30, f"the filter dropped {30 - kept_dark} hole-like detections"
    assert dropped_bright == 30, f"the filter kept {30 - dropped_bright} rope-like detections"


def test_it_is_off_by_default_in_the_inference_path():
    """A geometric prior tuned on synthetic damage must be opt-in.

    Real damage is not reliably dark — backscatter, a fish behind the hole, a
    partly-occluded tear — so switching this on globally would trade a measured
    false-alarm problem for an unmeasured missed-detection one.
    """
    import inspect

    from netinspect import inference

    src = inspect.getsource(inference)
    assert "polarity" not in src or "filter_detections" not in src, (
        "the polarity filter appears to be wired into NetInspector; it is a "
        "prior tuned on synthetic damage and must stay opt-in")

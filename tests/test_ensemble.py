"""Tests for the det-gated / seg-confirmed ensemble logic (no models needed)."""
from __future__ import annotations

from netinspect.ensemble import EnsembleConfig, combine
from netinspect.utils import BBox


def _b(x1, y1, x2, y2, score=0.9):
    return BBox(x1, y1, x2, y2, 0, "damage", score)


def test_agree_keeps_only_confirmed_boxes():
    det = [_b(10, 10, 50, 50), _b(200, 200, 240, 240)]   # two proposals
    seg = [_b(12, 12, 52, 52)]                            # confirms only the first
    out = combine(det, seg, EnsembleConfig(mode="agree", agree_iou=0.3))
    assert len(out) == 1
    assert out[0].x1 == 10  # geometry comes from the detector proposal


def test_agree_drops_all_when_seg_silent():
    det = [_b(10, 10, 50, 50)]
    out = combine(det, [], EnsembleConfig(mode="agree"))
    assert out == []


def test_conf_thresholds_filter_inputs():
    det = [_b(10, 10, 50, 50, score=0.2)]                # below det_conf
    seg = [_b(11, 11, 51, 51, score=0.9)]
    out = combine(det, seg, EnsembleConfig(mode="agree", det_conf=0.25))
    assert out == []


def test_modes_det_seg_union():
    det = [_b(10, 10, 50, 50)]
    seg = [_b(300, 300, 340, 340)]
    assert len(combine(det, seg, EnsembleConfig(mode="det"))) == 1
    assert len(combine(det, seg, EnsembleConfig(mode="seg"))) == 1
    assert len(combine(det, seg, EnsembleConfig(mode="union"))) == 2
    # disjoint boxes -> agreement keeps none
    assert len(combine(det, seg, EnsembleConfig(mode="agree"))) == 0

"""Tests for temporal tracking/confirmation and PatchCore config IO."""
from __future__ import annotations

import numpy as np
import pytest

from netinspect.temporal import TemporalConfig, Tracker, filter_sequence
from netinspect.utils import BBox


def _box(x, score=0.9):
    return BBox(x, x, x + 20, x + 20, 0, "damage", score)


def test_persistent_detection_is_confirmed_after_min_hits():
    tr = Tracker(TemporalConfig(min_hits=3, iou_match=0.2))
    # Same box for several frames (small jitter) -> confirmed at frame 3.
    out = []
    for i in range(5):
        out.append(tr.update([_box(50 + i)]))
    assert out[0] == [] and out[1] == []      # not yet confirmed
    assert len(out[2]) == 1                     # confirmed at the 3rd hit
    assert len(out[4]) == 1


def test_transient_detection_is_never_confirmed():
    tr = Tracker(TemporalConfig(min_hits=3, max_age=1))
    res = []
    res.append(tr.update([_box(50)]))           # one-frame blip
    res.append(tr.update([]))                    # gone
    res.append(tr.update([]))                    # aged out
    res.append(tr.update([]))
    assert all(r == [] for r in res)


def test_confidence_gate():
    tr = Tracker(TemporalConfig(min_hits=1, conf=0.5))
    assert tr.update([_box(10, score=0.4)]) == []   # below gate -> ignored
    assert len(tr.update([_box(10, score=0.9)])) == 1


def test_filter_sequence_reduces_transients():
    # 6 frames: a persistent box + a different transient box on frame 0 only.
    seq = [[_box(50 + i)] + ([_box(200)] if i == 0 else []) for i in range(6)]
    confirmed = filter_sequence(seq, TemporalConfig(min_hits=3))
    total_confirmed = sum(len(f) for f in confirmed)
    total_raw = sum(len(f) for f in seq)
    assert total_confirmed < total_raw          # transients removed
    assert any(len(f) for f in confirmed)       # persistent one kept


def test_patchcore_config_roundtrip(tmp_path):
    pytest.importorskip("torchvision")
    from netinspect.patchcore import PatchCoreConfig, PatchCoreModel
    cfg = PatchCoreConfig(coreset_size=10)
    model = PatchCoreModel(bank=np.random.rand(10, 384).astype(np.float32),
                           grid=(7, 7), threshold=2.5, cfg=cfg, train_stats={})
    model.save(tmp_path / "pc")
    loaded = PatchCoreModel.load(tmp_path / "pc")
    assert loaded.threshold == 2.5
    assert loaded.cfg.coreset_size == 10
    assert loaded.bank.shape == (10, 384)

"""Tests for the OOD gate calibration/flagging logic (no models needed)."""
from __future__ import annotations

import pytest

from netinspect.ood_gate import OODGate


def test_calibrate_sets_percentile_threshold():
    gate = OODGate.calibrate([1.0, 2.0, 3.0, 4.0, 100.0], percentile=80.0)
    # p80 of the five values is well above the bulk, below the outlier.
    assert 4.0 <= gate.threshold <= 100.0
    assert gate.flag(100.0) and not gate.flag(1.0)


def test_flag_rate():
    gate = OODGate(threshold=5.0)
    assert gate.flag_rate([1, 2, 3]) == 0.0
    assert gate.flag_rate([6, 7, 1, 2]) == 0.5
    assert gate.flag_rate([]) == 0.0


def test_calibrate_requires_scores():
    with pytest.raises(ValueError):
        OODGate.calibrate([])

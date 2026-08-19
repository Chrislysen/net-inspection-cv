"""Tests for the OOD gate calibration/flagging logic (no models needed)."""
from __future__ import annotations

import pytest

from netinspect.ood_gate import OODGate


def test_calibrate_sets_percentile_threshold():
    scores = [1.0, 2.0, 3.0, 4.0, 100.0]
    gate = OODGate.calibrate(scores, percentile=80.0)
    # The old bound was `4.0 <= threshold <= 100.0`, which a calibrate() that
    # ignored the percentile entirely (returning max, or the 4th value) also
    # satisfied. Pin it to the actual quantile instead.
    import numpy as np
    assert gate.threshold == pytest.approx(float(np.percentile(scores, 80.0)), rel=1e-6)
    assert gate.flag(100.0) and not gate.flag(1.0)

    # And it must MOVE with the percentile, which a constant cannot do.
    assert OODGate.calibrate(scores, percentile=99.0).threshold > gate.threshold
    assert OODGate.calibrate(scores, percentile=50.0).threshold < gate.threshold


def test_flag_rate():
    gate = OODGate(threshold=5.0)
    assert gate.flag_rate([1, 2, 3]) == 0.0
    assert gate.flag_rate([6, 7, 1, 2]) == 0.5
    assert gate.flag_rate([]) == 0.0


def test_calibrate_requires_scores():
    with pytest.raises(ValueError):
        OODGate.calibrate([])

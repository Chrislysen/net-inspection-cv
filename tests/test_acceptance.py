"""Tests for the release gate and threshold calibration.

The whole value of a gate is that it fails when it should. A gate that passes on
missing data is worse than no gate, because it launders an unvalidated model
into a deployment, so most of these tests are about refusing rather than
accepting.
"""
from __future__ import annotations

import pytest

from netinspect import acceptance as A
from netinspect.utils import BBox


def _box(score=0.9):
    return BBox(x1=10, y1=10, x2=20, y2=20, score=score, class_name="damage")


def _scenario(clean=100, damaged=100, false_alarms=2, detected=90, score=0.9):
    """Build predictions and ground truth with an exactly known outcome."""
    preds, gts = {}, {}
    for i in range(clean):
        name = f"clean_{i}.jpg"
        gts[name] = []
        preds[name] = [_box(score)] if i < false_alarms else []
    for i in range(damaged):
        name = f"dmg_{i}.jpg"
        gts[name] = [_box(1.0)]
        preds[name] = [_box(score)] if i < detected else []
    return preds, gts


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def test_rates_are_computed_against_their_own_denominators():
    preds, gts = _scenario(clean=200, damaged=50, false_alarms=10, detected=40)
    m = A.measure(preds, gts, conf=0.25)
    assert m["clean_frames"] == 200 and m["damaged_frames"] == 50
    assert m["false_alarm_rate"] == pytest.approx(10 / 200)
    assert m["recall"] == pytest.approx(40 / 50)


def test_a_rate_with_no_denominator_is_none_not_zero():
    """No clean frames means the false-alarm rate is unknown, not perfect."""
    preds, gts = _scenario(clean=0, damaged=20, detected=20)
    m = A.measure(preds, gts)
    assert m["false_alarm_rate"] is None
    assert m["recall"] == pytest.approx(1.0)


def test_no_damaged_frames_means_recall_is_unknown():
    preds, gts = _scenario(clean=20, damaged=0, false_alarms=1)
    assert A.measure(preds, gts)["recall"] is None


def test_raising_the_threshold_suppresses_low_confidence_alerts():
    preds, gts = _scenario(false_alarms=50, detected=100, score=0.4)
    assert A.measure(preds, gts, conf=0.3)["false_alarm_rate"] == pytest.approx(0.5)
    assert A.measure(preds, gts, conf=0.5)["false_alarm_rate"] == pytest.approx(0.0)
    assert A.measure(preds, gts, conf=0.5)["recall"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_a_good_model_passes():
    preds, gts = _scenario(clean=100, damaged=100, false_alarms=2, detected=90)
    v = A.gate(preds, gts, A.OperatingPoint(max_false_alarm_rate=0.05, min_recall=0.8))
    assert v.passed, v.summary()


def test_too_many_false_alarms_fails_even_with_perfect_recall():
    preds, gts = _scenario(clean=100, damaged=100, false_alarms=40, detected=100)
    v = A.gate(preds, gts, A.OperatingPoint(max_false_alarm_rate=0.05, min_recall=0.8))
    assert not v.passed
    assert any(c.name == "false alarm rate" and not c.passed for c in v.checks)


def test_poor_recall_fails_even_with_no_false_alarms():
    preds, gts = _scenario(clean=100, damaged=100, false_alarms=0, detected=10)
    v = A.gate(preds, gts, A.OperatingPoint(min_recall=0.8))
    assert not v.passed
    assert any(c.name == "recall" and not c.passed for c in v.checks)


def test_a_gate_with_no_clean_frames_fails_closed():
    """The dangerous case: nothing to measure false alarms on must never pass."""
    preds, gts = _scenario(clean=0, damaged=200, detected=200)
    v = A.gate(preds, gts)
    assert not v.passed
    fa = next(c for c in v.checks if c.name == "false alarm rate")
    assert fa.value is None and not fa.passed


def test_too_few_frames_fails_even_when_the_rates_look_perfect():
    preds, gts = _scenario(clean=4, damaged=4, false_alarms=0, detected=4)
    v = A.gate(preds, gts)
    assert not v.passed, "a gate over 8 frames must not certify anything"
    assert any(c.name == "clean frames" and not c.passed for c in v.checks)


def test_an_empty_evaluation_set_cannot_pass():
    v = A.gate({}, {})
    assert not v.passed


def test_precision_is_only_checked_when_the_contract_asks_for_it():
    preds, gts = _scenario(clean=100, damaged=100, false_alarms=2, detected=90)
    assert not any(c.name == "precision" for c in A.gate(preds, gts).checks)
    op = A.OperatingPoint(min_precision=0.99)
    assert any(c.name == "precision" for c in A.gate(preds, gts, op).checks)


def test_the_summary_states_the_decision_and_the_numbers():
    preds, gts = _scenario(clean=100, damaged=100, false_alarms=40, detected=100)
    text = A.gate(preds, gts).summary()
    assert "FAIL" in text and "must not be deployed" in text
    assert "false alarm rate" in text


def test_the_operating_point_is_recorded_with_the_verdict():
    preds, gts = _scenario()
    op = A.OperatingPoint(conf=0.4, max_false_alarm_rate=0.02)
    d = A.gate(preds, gts, op).to_dict()
    assert d["operating_point"]["conf"] == 0.4
    assert d["operating_point"]["max_false_alarm_rate"] == 0.02


def test_a_verdict_can_be_written_and_reread(tmp_path):
    import json
    preds, gts = _scenario()
    p = A.write_verdict(A.gate(preds, gts), tmp_path / "verdict.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "passed" in d and "checks" in d and "summary" in d


def test_operating_point_round_trips_and_ignores_unknown_keys():
    op = A.OperatingPoint.from_dict({"conf": 0.33, "min_recall": 0.6, "nonsense": 1})
    assert op.conf == 0.33 and op.min_recall == 0.6


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _graded():
    """Alerts whose confidence separates clean from damaged, so a threshold exists."""
    preds, gts = {}, {}
    for i in range(100):
        name = f"clean_{i}.jpg"
        gts[name] = []
        preds[name] = [_box(0.30)]            # every clean frame alerts, but weakly
    for i in range(100):
        name = f"dmg_{i}.jpg"
        gts[name] = [_box(1.0)]
        preds[name] = [_box(0.80)]            # damage alerts strongly
    return preds, gts


def test_calibration_finds_a_threshold_that_meets_the_budget():
    out = A.choose_threshold(*_graded(), target_false_alarm_rate=0.05)
    assert out["achievable"]
    assert out["chosen"]["false_alarm_rate"] <= 0.05
    assert out["chosen"]["recall"] == pytest.approx(1.0)
    assert 0.30 < out["chosen"]["conf"] <= 0.80


def test_calibration_picks_the_lowest_feasible_threshold_to_keep_recall():
    out = A.choose_threshold(*_graded(), target_false_alarm_rate=0.05)
    feasible = [r for r in out["sweep"]
                if r["false_alarm_rate"] is not None and r["false_alarm_rate"] <= 0.05]
    assert out["chosen"]["conf"] == min(r["conf"] for r in feasible)


def test_calibration_says_so_when_no_threshold_can_meet_the_budget():
    preds, gts = {}, {}
    for i in range(50):
        gts[f"clean_{i}.jpg"] = []
        preds[f"clean_{i}.jpg"] = [_box(0.99)]      # fires hard on clean net
    for i in range(50):
        gts[f"dmg_{i}.jpg"] = [_box(1.0)]
        preds[f"dmg_{i}.jpg"] = [_box(0.99)]
    out = A.choose_threshold(preds, gts, target_false_alarm_rate=0.01)
    assert not out["achievable"] and out["chosen"] is None
    assert "No threshold" in out["note"]


def test_calibration_warns_that_a_threshold_does_not_transfer():
    out = A.choose_threshold(*_graded(), target_false_alarm_rate=0.05)
    assert "recalibrate" in out["note"].lower()


def test_the_sweep_is_returned_so_the_tradeoff_is_visible():
    out = A.choose_threshold(*_graded())
    assert len(out["sweep"]) > 5
    assert all("false_alarm_rate" in r and "recall" in r for r in out["sweep"])

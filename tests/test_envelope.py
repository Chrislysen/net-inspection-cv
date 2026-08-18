"""Tests for the inspection operating envelope: statistics, gate, and fitting.

No models, bags or telemetry files are needed — this is all pure logic, which
is why it runs in CI where rosbags/ultralytics are deliberately not installed.
"""
from __future__ import annotations

import math

import pytest

from netinspect.envelope import (
    IN_ENVELOPE,
    OUT_OF_ENVELOPE,
    UNKNOWN,
    EnvelopeGate,
    EnvelopeSpec,
    dose_response,
    fit_envelope,
    intervals_overlap,
    matched_band_comparison,
    proportion_stat,
    wilson_ci,
)


# --------------------------------------------------------------------------- #
# Binomial statistics
# --------------------------------------------------------------------------- #
def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(5, 100)
    assert lo < 0.05 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_ci_stays_inside_unit_interval_at_zero_events():
    # The Wald interval would give [0, 0] here, implying certainty from 10 frames.
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi > 0.0, "zero events in a small sample must not imply a zero rate"


def test_wilson_ci_narrows_with_more_data():
    _, hi_small = wilson_ci(0, 10)
    _, hi_large = wilson_ci(0, 1000)
    assert hi_large < hi_small


def test_wilson_ci_empty_sample_is_nan():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_ci_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_proportion_stat_counts_and_rate():
    s = proportion_stat([1, 0, 0, 1, 0])
    assert s["n"] == 5 and s["k"] == 2 and s["rate"] == 0.4
    assert s["ci95"][0] <= 0.4 <= s["ci95"][1]


def test_proportion_stat_empty():
    s = proportion_stat([])
    assert s["n"] == 0 and s["rate"] is None and s["ci95"] is None


def test_intervals_overlap():
    assert intervals_overlap([0.0, 0.2], [0.1, 0.3])
    assert not intervals_overlap([0.0, 0.05], [0.10, 0.30])
    assert intervals_overlap(None, [0.1, 0.2])  # missing data -> not a claim of difference


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def _spec():
    return EnvelopeSpec(standoff_min_m=0.4, standoff_max_m=0.9, speed_max_ms=0.2)


def test_frame_inside_the_envelope_is_trusted():
    v = EnvelopeGate(_spec()).check(standoff_m=0.6, speed_ms=0.1, locked=True)
    assert v.status == IN_ENVELOPE and v.trusted and v.reasons == []


def test_frame_beyond_validated_standoff_is_flagged():
    v = EnvelopeGate(_spec()).check(standoff_m=1.4, speed_ms=0.1, locked=True)
    assert v.status == OUT_OF_ENVELOPE and not v.trusted
    assert any("above validated maximum" in r for r in v.reasons)


def test_frame_too_close_is_also_flagged():
    v = EnvelopeGate(_spec()).check(standoff_m=0.15, speed_ms=0.1, locked=True)
    assert v.status == OUT_OF_ENVELOPE
    assert any("below validated minimum" in r for r in v.reasons)


def test_excess_sweep_speed_is_flagged_independently():
    v = EnvelopeGate(_spec()).check(standoff_m=0.6, speed_ms=0.35, locked=True)
    assert v.status == OUT_OF_ENVELOPE
    assert any("sweep speed" in r for r in v.reasons)


def test_multiple_violations_are_all_reported():
    v = EnvelopeGate(_spec()).check(standoff_m=1.4, speed_ms=0.35, locked=True)
    assert len(v.reasons) == 2


def test_missing_telemetry_is_unknown_not_trusted():
    """Absence of evidence must never be read as compliance."""
    gate = EnvelopeGate(_spec())
    assert gate.check(standoff_m=None, locked=True).status == UNKNOWN
    assert gate.check(standoff_m=float("nan"), locked=True).status == UNKNOWN
    assert not gate.check(standoff_m=None, locked=True).trusted


def test_unlocked_net_plane_estimate_is_unknown():
    v = EnvelopeGate(_spec()).check(standoff_m=0.6, speed_ms=0.1, locked=False)
    assert v.status == UNKNOWN
    assert not v.trusted


def test_lock_requirement_can_be_disabled():
    spec = EnvelopeSpec(standoff_min_m=0.4, standoff_max_m=0.9, require_lock=False)
    assert EnvelopeGate(spec).check(standoff_m=0.6, locked=False).status == IN_ENVELOPE


def test_summarise_counts_unknown_against_compliance():
    gate = EnvelopeGate(_spec())
    verdicts = [
        gate.check(standoff_m=0.6, speed_ms=0.1, locked=True),   # in
        gate.check(standoff_m=0.6, speed_ms=0.1, locked=True),   # in
        gate.check(standoff_m=1.4, speed_ms=0.1, locked=True),   # out
        gate.check(standoff_m=None, locked=True),                 # unknown
    ]
    s = gate.summarise(verdicts)
    assert s["frames"] == 4
    assert s["counts"][IN_ENVELOPE] == 2
    assert s["counts"][OUT_OF_ENVELOPE] == 1
    assert s["counts"][UNKNOWN] == 1
    assert s["compliance"] == 0.5
    assert "re-fly" in s["verdict"]


def test_summarise_accepts_a_compliant_inspection():
    gate = EnvelopeGate(_spec())
    verdicts = [gate.check(standoff_m=0.6, speed_ms=0.1, locked=True) for _ in range(20)]
    s = gate.summarise(verdicts)
    assert s["compliance"] == 1.0
    assert s["verdict"].startswith("accept")


def test_spec_roundtrips_through_dict():
    spec = EnvelopeSpec(standoff_min_m=0.4, standoff_max_m=0.9, speed_max_ms=0.2,
                        model="det_v1", evidence={"frames": 100})
    back = EnvelopeSpec.from_dict(spec.to_dict())
    assert back.standoff_min_m == 0.4 and back.standoff_max_m == 0.9
    assert back.model == "det_v1"


def test_unbounded_spec_roundtrips_without_inf_in_json():
    spec = EnvelopeSpec(standoff_min_m=0.0)
    d = spec.to_dict()
    assert d["standoff_max_m"] is None          # JSON-safe
    assert math.isinf(EnvelopeSpec.from_dict(d).standoff_max_m)


# --------------------------------------------------------------------------- #
# Dose-response and fitting
# --------------------------------------------------------------------------- #
def test_dose_response_bins_and_skips_empty_bins():
    standoff = [0.5, 0.55, 0.9, 0.95]
    events = [0, 0, 1, 1]
    rows = dose_response(standoff, events, [0.4, 0.6, 0.8, 1.0])
    assert [r["standoff_lo"] for r in rows] == [0.4, 0.8]   # 0.6-0.8 is empty, omitted
    assert rows[0]["rate"] == 0.0 and rows[1]["rate"] == 1.0


def test_dose_response_ignores_missing_standoff():
    rows = dose_response([0.5, float("nan"), 0.55], [0, 1, 0], [0.4, 0.6])
    assert rows[0]["n"] == 2


def test_fit_envelope_picks_the_clean_low_standoff_range():
    # Clean up to 0.8 m, then false alarms climb.
    standoff = [0.45] * 100 + [0.65] * 100 + [1.05] * 100
    events = [0] * 100 + [0] * 100 + [1] * 60 + [0] * 40
    rows = dose_response(standoff, events, [0.4, 0.6, 0.8, 1.0, 1.2])
    spec = fit_envelope(rows, target_rate=0.05, model="det_v1")
    assert spec.evidence["fitted"] is True
    assert spec.standoff_min_m == 0.4
    assert spec.standoff_max_m == 0.8      # stops before the 1.0-1.2 bin
    assert spec.model == "det_v1"


def test_fit_envelope_requires_upper_ci_not_just_point_estimate():
    """Three clean frames must not be enough to certify a bin."""
    rows = dose_response([0.45, 0.46, 0.47], [0, 0, 0], [0.4, 0.6])
    spec = fit_envelope(rows, target_rate=0.05)
    assert spec.evidence["fitted"] is False
    assert math.isnan(spec.standoff_min_m)


def test_fit_envelope_reports_failure_rather_than_permissive_default():
    standoff = [0.5] * 100
    events = [1] * 100                     # everything false-alarms
    rows = dose_response(standoff, events, [0.4, 0.6])
    spec = fit_envelope(rows, target_rate=0.05)
    assert spec.evidence["fitted"] is False
    assert "No standoff bin" in spec.evidence["note"]
    assert not EnvelopeGate(spec).check(standoff_m=0.5, locked=True).trusted


def test_fit_envelope_records_its_evidence_and_caveat():
    rows = dose_response([0.5] * 200, [0] * 200, [0.4, 0.6])
    spec = fit_envelope(rows, target_rate=0.05)
    assert spec.evidence["frames"] == 200
    assert spec.evidence["measured_rate"] == 0.0
    assert "unvalidated" in spec.evidence["caveat"]


# --------------------------------------------------------------------------- #
# Matched-band comparison
# --------------------------------------------------------------------------- #
def test_matched_band_overlapping_rates_do_not_claim_a_group_effect():
    out = matched_band_comparison({"day_a": [0, 0, 1, 0] * 10, "day_b": [0, 1, 0, 0] * 10})
    assert out["intervals_overlap"] is True
    assert "not needed to explain" in out["interpretation"]


def test_matched_band_separated_rates_report_a_residual_effect():
    out = matched_band_comparison({"day_a": [0] * 100, "day_b": [1] * 100})
    assert out["intervals_overlap"] is False
    assert "residual effect" in out["interpretation"]


def test_matched_band_handles_an_empty_group():
    out = matched_band_comparison({"day_a": [0, 1, 0], "day_b": []})
    assert out["groups"]["day_b"]["n"] == 0
    assert "interpretation" not in out      # no comparison claimed from nothing

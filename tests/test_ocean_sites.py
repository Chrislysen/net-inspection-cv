"""Tests for the site registry, ocean conditions and cod thermal physics.

No network: every test here exercises pure logic against constructed inputs, so
the suite stays runnable in CI where the Fiskeridirektoratet and MET APIs are
not reachable.
"""
from __future__ import annotations

import math

import pytest

from netinspect.ocean import (
    SWEEP_SPEED_RANGE_MS,
    OceanSample,
    best_window,
    degree_days,
    inspection_windows,
    rate_conditions,
    summarise_windows,
    tgc_growth,
    tgc_growth_thermal_corrected,
    thermal_optimum_c,
    thermal_stress,
)
from netinspect.sites import UNIT_COUNT, UNIT_TONNES, Site, by_operator, summarise


# --------------------------------------------------------------------------- #
# Site registry
# --------------------------------------------------------------------------- #
def _site(**kw):
    base = dict(loknr=1, name="X", status="AKTIV", lat=62.0, lon=6.0,
                county="MØRE OG ROMSDAL", municipality="ÅLESUND", capacity=3599.0,
                capacity_unit=UNIT_TONNES, placement="SJØ", species="Torsk",
                operators="ODE AS", permits="M-X-0001", production_form="Matfisk")
    base.update(kw)
    return Site(**base)


def test_sea_site_detection_handles_norwegian_placement():
    assert _site(placement="SJØ").is_sea_site
    assert _site(placement="SJO").is_sea_site
    assert not _site(placement="LAND").is_sea_site


def test_only_sea_sites_have_a_net_to_inspect():
    land = _site(placement="LAND", capacity=16_900_000, capacity_unit=UNIT_COUNT)
    assert not land.is_sea_site


def test_capacity_tonnes_is_none_for_fish_counts():
    """Tonnes and individuals are different units and must not be conflated."""
    assert _site(capacity=3599.0, capacity_unit=UNIT_TONNES).capacity_tonnes == 3599.0
    assert _site(capacity=1e7, capacity_unit=UNIT_COUNT).capacity_tonnes is None


def test_by_operator_is_case_insensitive_substring():
    pool = [_site(loknr=1, operators="ODE AS"),
            _site(loknr=2, operators="ODE AS, ODE PROCESSING AS"),
            _site(loknr=3, operators="NORCOD AS")]
    assert {s.loknr for s in by_operator("ode", pool)} == {1, 2}
    assert {s.loknr for s in by_operator("NORCOD", pool)} == {3}


def test_by_operator_matches_every_holder_on_a_shared_site():
    pool = [_site(loknr=9, operators="ODE AS, ODE PROCESSING AS")]
    assert by_operator("ODE PROCESSING", pool)
    assert by_operator("ODE AS", pool)


def test_summarise_separates_tonnage_from_individuals():
    pool = [_site(loknr=1, capacity=3599.0, capacity_unit=UNIT_TONNES, placement="SJØ"),
            _site(loknr=2, capacity=1560.0, capacity_unit=UNIT_TONNES, placement="SJØ"),
            _site(loknr=3, capacity=1.69e7, capacity_unit=UNIT_COUNT, placement="LAND")]
    s = summarise(pool)
    assert s["sea_sites"] == 2 and s["land_sites"] == 1
    assert s["licensed_tonnes_total"] == pytest.approx(5159.0)
    assert s["licensed_individuals_total"] == 16_900_000


def test_summarise_counts_counties():
    pool = [_site(county="VESTLAND"), _site(county="VESTLAND"),
            _site(county="TRØNDELAG")]
    assert summarise(pool)["counties"]["VESTLAND"] == 2


# --------------------------------------------------------------------------- #
# Inspection windows
# --------------------------------------------------------------------------- #
def _sample(wave=0.2, cur=0.05):
    return OceanSample(time="2026-08-18T14:00:00Z",
                       sea_surface_wave_height_m=wave, sea_water_speed_ms=cur,
                       sea_water_temperature_c=13.5)


def test_calm_conditions_rate_good():
    w = rate_conditions(_sample(wave=0.2, cur=0.03))
    assert w.rating == "good"


def test_high_waves_rate_poor():
    w = rate_conditions(_sample(wave=1.6, cur=0.03))
    assert w.rating == "poor"
    assert any("wave height" in r for r in w.reasons)


def test_current_at_sweep_speed_degrades_the_rating():
    """A current matching the commanded sweep is not a minor disturbance."""
    w = rate_conditions(_sample(wave=0.2, cur=SWEEP_SPEED_RANGE_MS[1]))
    assert w.rating == "poor"
    assert any("net-following" in r for r in w.reasons)


def test_current_near_slowest_sweep_is_marginal():
    w = rate_conditions(_sample(wave=0.2, cur=SWEEP_SPEED_RANGE_MS[0] + 0.01))
    assert w.rating == "marginal"


def test_current_is_reported_relative_to_sweep_speed():
    w = rate_conditions(_sample(cur=0.20))
    assert w.current_vs_sweep == pytest.approx(2.0, abs=0.01)


def test_missing_forecast_values_are_unknown_not_good():
    w = rate_conditions(OceanSample(time="t"))
    assert w.rating == "unknown"


def test_worst_factor_decides_the_rating():
    w = rate_conditions(_sample(wave=0.2, cur=0.4))
    assert w.rating == "poor"


def test_best_window_prefers_good_then_marginal():
    ws = inspection_windows([_sample(wave=1.6), _sample(wave=0.7), _sample(wave=0.1)])
    assert best_window(ws).rating == "good"

    only_marginal = inspection_windows([_sample(wave=1.6), _sample(wave=0.7)])
    assert best_window(only_marginal).rating == "marginal"


def test_best_window_returns_none_when_nothing_is_workable():
    assert best_window(inspection_windows([_sample(wave=2.0)])) is None


def test_summarise_windows_reports_counts_and_caveat():
    s = summarise_windows(inspection_windows([_sample(wave=0.1)] * 3
                                             + [_sample(wave=1.6)]))
    assert s["hours"] == 4
    assert s["ratings"]["good"] == 3
    assert s["good_fraction"] == 0.75
    assert "not certified" in s["caveat"]


def test_summarise_windows_empty():
    assert summarise_windows([])["hours"] == 0


# --------------------------------------------------------------------------- #
# Cod thermal physics
# --------------------------------------------------------------------------- #
def test_thermal_optimum_falls_as_the_fish_grows():
    """The central non-obvious fact: bigger cod want colder water."""
    assert thermal_optimum_c(2) > thermal_optimum_c(200) > thermal_optimum_c(4000)


def test_thermal_optimum_matches_published_values():
    assert thermal_optimum_c(2000) == pytest.approx(9.17, abs=0.05)   # 2 kg
    assert thermal_optimum_c(2) == pytest.approx(14.99, abs=0.05)     # 2 g


def test_thermal_optimum_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        thermal_optimum_c(0)


def test_norwegian_summer_is_above_optimum_for_market_size_cod():
    """Warm water that suits a hatchery is a problem for a grow-out pen."""
    s = thermal_stress(2000, 13.5)
    assert s["delta_c"] > 3
    assert "above optimum" in s["state"]


def test_same_temperature_suits_a_juvenile():
    assert thermal_stress(2, 13.5)["state"] in ("near optimum", "slightly below optimum")


def test_thermal_stress_reports_near_optimum():
    assert thermal_stress(2000, 9.2)["state"] == "near optimum"


def test_tgc_growth_is_monotonic_in_warm_water():
    w = tgc_growth(1000, [10.0] * 30)
    assert w[-1] > w[0] > 1000
    assert all(b >= a for a, b in zip(w, w[1:]))


def test_tgc_growth_ignores_freezing_temperatures():
    assert tgc_growth(1000, [-2.0] * 10)[-1] == pytest.approx(1000.0)


def test_tgc_growth_rejects_nonpositive_start():
    with pytest.raises(ValueError):
        tgc_growth(0, [10.0])


def test_uncorrected_tgc_overpredicts_growth_in_warm_water():
    """The bias the corrected form exists to expose.

    A 2 kg cod at 15 degC is well above its ~9 degC optimum. Plain TGC treats
    the extra warmth as extra growth; the corrected form does not.
    """
    plain = tgc_growth(2000, [15.0] * 90)[-1]
    corrected = tgc_growth_thermal_corrected(2000, [15.0] * 90)[-1]
    assert plain > corrected


def test_the_two_models_agree_near_the_optimum():
    opt = thermal_optimum_c(2000)
    plain = tgc_growth(2000, [opt] * 30)[-1]
    corrected = tgc_growth_thermal_corrected(2000, [opt] * 30)[-1]
    assert corrected == pytest.approx(plain, rel=0.02)


def test_degree_days_accumulates_positive_temperature_only():
    assert degree_days([10.0] * 10) == pytest.approx(100.0)
    assert degree_days([-5.0] * 10) == pytest.approx(0.0)


def test_degree_days_scales_with_step_length():
    assert degree_days([10.0] * 4, days_per_step=0.25) == pytest.approx(10.0)


def test_growth_series_length_matches_input():
    assert len(tgc_growth(1000, [8.0] * 17)) == 17
    assert len(tgc_growth_thermal_corrected(1000, [8.0] * 17)) == 17


def test_thermal_correction_recomputes_optimum_as_fish_grows():
    """Efficiency must track the fish's changing optimum, not a fixed one."""
    cold = tgc_growth_thermal_corrected(50, [12.0] * 400)
    assert math.isfinite(cold[-1]) and cold[-1] > 50

"""Tests for the Jerlov underwater degradation model.

The model is physics, so the tests are physics: red must go first, deeper must
mean worse, and turbid coastal water must degrade more than clear ocean. A bug
here would produce plausible-looking green images that encode the wrong
attenuation, which no visual check would catch.
"""
from __future__ import annotations

import numpy as np
import pytest

from netinspect import water as W


def _frame(value=180, size=16):
    return np.full((size, size, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# the physics
# --------------------------------------------------------------------------- #
def test_red_dies_first_until_scattering_takes_over():
    """Absorption removes red first — but only while absorption dominates.

    In the most turbid coastal water, scattering strips the short wavelengths
    faster than absorption strips the long ones and the ordering INVERTS. That
    is why open ocean is blue, a fjord is green, and harbour water is brown.
    """
    for name in ("I", "IA", "IB", "II", "III", "1C", "3C", "5C"):
        r, g, b = W.WATER_TYPES[name]
        assert r < g, f"{name}: red must attenuate faster than green"
        assert r <= b, f"{name}: red must not outlast blue while absorption dominates"

    for name in ("7C", "9C"):
        r, g, b = W.WATER_TYPES[name]
        assert b < g <= r, (
            f"{name}: in the murkiest water scattering should invert the order, "
            "leaving blue the worst-transmitted channel")


def test_clear_ocean_transmits_blue_best():
    for name in ("I", "IA", "IB", "II"):
        _, g, b = W.WATER_TYPES[name]
        assert b > g, f"{name}: clear ocean should transmit blue best"


def test_overall_transmission_falls_monotonically_along_the_series():
    """The classification is ordered by clarity, so the mean must be too."""
    order = ["I", "IA", "IB", "II", "III", "1C", "3C", "5C", "7C", "9C"]
    means = [sum(W.WATER_TYPES[n]) / 3 for n in order]
    assert means == sorted(means, reverse=True), f"out of order: {means}"


def test_turbid_coastal_water_attenuates_more_than_clear_ocean():
    clear = np.mean(W.WATER_TYPES["I"])
    turbid = np.mean(W.WATER_TYPES["9C"])
    assert turbid < clear


def test_transmission_falls_with_distance():
    near = W.WaterConfig(water_type="5C", distance_m=0.5).transmission()
    far = W.WaterConfig(water_type="5C", distance_m=4.0).transmission()
    assert np.all(far < near)


def test_zero_distance_transmits_everything():
    t = W.WaterConfig(water_type="9C", distance_m=0.0).transmission()
    assert np.allclose(t, 1.0)


def test_an_unknown_water_type_is_refused_with_the_valid_list():
    with pytest.raises(ValueError) as e:
        W.WaterConfig(water_type="Atlantis").transmission()
    assert "5C" in str(e.value)


# --------------------------------------------------------------------------- #
# applying it
# --------------------------------------------------------------------------- #
def test_applying_water_shifts_a_grey_frame_away_from_red():
    out = W.apply_water(_frame(), W.WaterConfig(water_type="5C", distance_m=2.0))
    r, g, b = (float(out[..., i].mean()) for i in range(3))
    assert r < g, "red must be the most depleted channel"


def test_more_turbid_water_degrades_more():
    frame = _frame()
    clear = W.apply_water(frame, W.WaterConfig(water_type="I", distance_m=2.0))
    murky = W.apply_water(frame, W.WaterConfig(water_type="9C", distance_m=2.0))
    # Contrast against the original collapses as turbidity rises.
    assert (np.abs(murky.astype(int) - frame.astype(int)).mean()
            > np.abs(clear.astype(int) - frame.astype(int)).mean())


def test_strength_zero_leaves_the_frame_untouched():
    frame = _frame()
    out = W.apply_water(frame, W.WaterConfig(water_type="9C", distance_m=5.0, strength=0.0))
    assert np.array_equal(out, frame), "strength=0 must be a no-op"


def test_output_stays_a_valid_uint8_image():
    out = W.apply_water(_frame(250), W.WaterConfig(water_type="9C", distance_m=10.0))
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


def test_shape_is_preserved():
    frame = np.zeros((7, 13, 3), dtype=np.uint8)
    assert W.apply_water(frame).shape == frame.shape


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def test_random_water_only_samples_deployable_conditions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cfg = W.random_water(rng)
        assert cfg.water_type in W.COASTAL_TYPES, (
            "sampling open-ocean types would train for water a farm ROV never sees")
        assert 0.0 <= cfg.strength <= 1.0


def test_augmentation_probability_is_honoured():
    rng = np.random.default_rng(1)
    frame = _frame()
    never = [W.augment(frame, rng, probability=0.0) for _ in range(20)]
    assert all(np.array_equal(f, frame) for f in never)

    rng = np.random.default_rng(1)
    always = [W.augment(frame, rng, probability=1.0) for _ in range(20)]
    assert any(not np.array_equal(f, frame) for f in always)


def test_augmentation_is_deterministic_for_a_seed():
    a = W.augment(_frame(), np.random.default_rng(7), probability=1.0)
    b = W.augment(_frame(), np.random.default_rng(7), probability=1.0)
    assert np.array_equal(a, b)

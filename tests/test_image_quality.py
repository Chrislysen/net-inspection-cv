"""Tests for the no-reference capture-quality metrics and quality banding.

Sharpness is the measurement that turned out to matter: it is the strongest
per-frame correlate of false alarms in this project, with a sign that differs
between models. So the properties worth pinning down are the ones an analysis
would silently rely on — that sharpness actually falls with blur, that a band
fitted on too little data refuses to certify itself, and that the OpenCV path
and the pure-numpy fallback agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from netinspect.image_quality import (
    DARK_LEVEL,
    METRIC_NAMES,
    QualityBand,
    _luminance,
    compute,
    fit_band,
    variance_of_laplacian,
)


def _checkerboard(size=64, cell=4):
    """A high-frequency pattern: maximally sharp for a focus measure."""
    ys, xs = np.mgrid[0:size, 0:size]
    grid = (((ys // cell) + (xs // cell)) % 2 * 255).astype(np.uint8)
    return np.dstack([grid] * 3)


def _flat(size=64, value=128):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _blur(img, passes=6):
    """Cheap box blur, so the test does not depend on scipy or cv2."""
    out = img.astype(np.float32)
    for _ in range(passes):
        out = (out
               + np.roll(out, 1, 0) + np.roll(out, -1, 0)
               + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5.0
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Sharpness
# --------------------------------------------------------------------------- #
def test_sharpness_falls_with_blur():
    """The property the whole analysis leans on."""
    sharp = variance_of_laplacian(_checkerboard())
    blurred = variance_of_laplacian(_blur(_checkerboard()))
    assert sharp > blurred


def test_flat_image_has_almost_no_high_frequency_content():
    assert variance_of_laplacian(_flat()) == pytest.approx(0.0, abs=1e-6)


def test_sharpness_is_nonnegative():
    for img in (_flat(), _checkerboard(), _blur(_checkerboard())):
        assert variance_of_laplacian(img) >= 0.0


def test_opencv_and_numpy_paths_agree(monkeypatch):
    """The fallback exists for minimal installs; it must not shift the metric."""
    img = _checkerboard()
    with_cv2 = variance_of_laplacian(img)
    monkeypatch.setattr("netinspect.image_quality.optional_import",
                        lambda name: None)
    without_cv2 = variance_of_laplacian(img)
    assert without_cv2 == pytest.approx(with_cv2, rel=0.05)


def test_grayscale_input_is_accepted():
    gray = _checkerboard()[:, :, 0]
    assert variance_of_laplacian(gray) > 0


# --------------------------------------------------------------------------- #
# The full metric set
# --------------------------------------------------------------------------- #
def test_compute_returns_every_declared_metric():
    m = compute(_checkerboard()).to_dict()
    assert set(m) == set(METRIC_NAMES)
    assert all(isinstance(v, float) for v in m.values())


def test_brightness_and_contrast_behave():
    dark = compute(_flat(value=20))
    bright = compute(_flat(value=230))
    assert dark.brightness < bright.brightness
    assert compute(_flat()).contrast == pytest.approx(0.0, abs=1e-6)
    assert compute(_checkerboard()).contrast > 100


def test_dark_fraction_counts_pixels_below_the_threshold():
    assert compute(_flat(value=DARK_LEVEL - 10)).dark_fraction == pytest.approx(1.0)
    assert compute(_flat(value=DARK_LEVEL + 10)).dark_fraction == pytest.approx(0.0)


def test_saturation_is_zero_for_grey_and_positive_for_colour():
    assert compute(_flat()).saturation == pytest.approx(0.0)
    colour = np.zeros((16, 16, 3), dtype=np.uint8)
    colour[..., 0] = 200
    assert compute(colour).saturation > 0


def test_luminance_uses_rec601_weights():
    green = np.zeros((4, 4, 3), dtype=np.uint8)
    green[..., 1] = 255
    blue = np.zeros((4, 4, 3), dtype=np.uint8)
    blue[..., 2] = 255
    assert _luminance(green).mean() > _luminance(blue).mean()


def test_metrics_round_trip_to_plain_floats():
    d = compute(_checkerboard()).to_dict()
    import json
    assert json.loads(json.dumps(d)) == d


# --------------------------------------------------------------------------- #
# Band fitting
# --------------------------------------------------------------------------- #
def test_band_fits_the_clean_low_sharpness_range():
    # Each quantile bin needs ~72 clean samples before a 5% target is
    # reachable at 95% confidence — the module is deliberately conservative.
    values = np.concatenate([np.linspace(40, 70, 800), np.linspace(350, 450, 800)])
    events = np.concatenate([np.zeros(800, int), np.ones(800, int)])
    band = fit_band(values, events, target_rate=0.05, model="det_v1")
    assert band.evidence["fitted"] is True
    assert band.model == "det_v1"
    assert band.evidence["measured_rate"] == 0.0


def test_band_refuses_to_certify_when_nothing_qualifies():
    values = np.linspace(10, 500, 200)
    events = np.ones(200, int)          # everything false-alarms
    band = fit_band(values, events, target_rate=0.05)
    assert band.evidence["fitted"] is False
    assert band.sharpness_min is None and band.sharpness_max is None


def test_band_requires_the_upper_interval_not_the_point_estimate():
    """A handful of clean frames must not certify a range."""
    band = fit_band(np.array([50.0, 51.0, 52.0]), np.zeros(3, int), target_rate=0.05)
    assert band.evidence["fitted"] is False


def test_band_handles_empty_input():
    band = fit_band(np.array([]), np.array([]), target_rate=0.05)
    assert band.evidence["fitted"] is False


def test_band_handles_a_constant_metric():
    band = fit_band(np.full(100, 42.0), np.zeros(100, int), target_rate=0.05)
    assert band.evidence["fitted"] is False
    assert "constant" in band.evidence["note"]


def test_band_ignores_non_finite_values():
    """NaN/inf must be dropped, not propagated into the quantile edges."""
    clean = np.linspace(40.0, 90.0, 800)
    values = np.concatenate([clean, np.array([np.nan, np.inf])])
    events = np.concatenate([np.zeros(800, int), np.array([1, 1])])
    band = fit_band(values, events, target_rate=0.05)
    assert band.evidence["fitted"] is True
    assert band.evidence["frames"] == 800


def test_fitted_band_carries_its_caveat():
    band = fit_band(np.linspace(40.0, 90.0, 800), np.zeros(800, int),
                    target_rate=0.05)
    assert band.evidence["fitted"] is True
    assert "unmeasured" in band.evidence["caveat"]


# --------------------------------------------------------------------------- #
# Band checking
# --------------------------------------------------------------------------- #
def _metrics(sharpness=100.0, contrast=30.0):
    from netinspect.image_quality import QualityMetrics
    return QualityMetrics(sharpness=sharpness, contrast=contrast, brightness=110.0,
                          saturation=10.0, dark_fraction=0.02)


def test_frame_inside_the_band_passes():
    ok, reasons = QualityBand(sharpness_min=50, sharpness_max=200).check(_metrics())
    assert ok and reasons == []


def test_blurred_frame_is_flagged_with_a_reason():
    ok, reasons = QualityBand(sharpness_min=50).check(_metrics(sharpness=10))
    assert not ok
    assert any("blur" in r for r in reasons)


def test_over_sharp_frame_is_flagged():
    """Too much fine structure is its own failure mode for the detector."""
    ok, reasons = QualityBand(sharpness_max=200).check(_metrics(sharpness=400))
    assert not ok
    assert any("fine structure" in r for r in reasons)


def test_contrast_bounds_are_checked_independently():
    ok, reasons = QualityBand(contrast_min=40).check(_metrics(contrast=20))
    assert not ok and any("contrast" in r for r in reasons)


def test_an_unbounded_band_accepts_everything():
    ok, _ = QualityBand().check(_metrics(sharpness=99999))
    assert ok


def test_band_round_trips_through_dict():
    band = QualityBand(sharpness_min=50, sharpness_max=200, model="det_v1")
    back = QualityBand.from_dict(band.to_dict())
    assert back.sharpness_min == 50 and back.model == "det_v1"


def test_from_dict_ignores_unknown_keys():
    band = QualityBand.from_dict({"sharpness_min": 10, "not_a_field": 1})
    assert band.sharpness_min == 10

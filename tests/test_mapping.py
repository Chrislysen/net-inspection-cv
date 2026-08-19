"""Tests for damage localisation and mapping.

The geometry here is the kind that looks right in a figure while being wrong in
metres, so these tests check invariants rather than eyeballed numbers: a
synthetic pass with a *known* answer, sign conventions that must agree with each
other, and the failure modes that should degrade honestly rather than silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from netinspect import mapping as M


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _motions(n, dx=-40.0, dy=0.0, inliers=200, matches=250):
    """`n` frame-to-frame motions. Negative dx = scene slides left = moving +along."""
    return [M.FrameMotion(index=i, dx_px=dx, dy_px=dy, rotation_deg=0.0,
                          inliers=inliers, matches=matches)
            for i in range(n)]


def _calibrate(n=10, dx=-40.0, standoff=1.0, travel_m=0.80, width=None):
    """Calibrate from `n` pairs whose totals are known exactly."""
    mot = _motions(n, dx=dx)
    return M.calibrate_scale([m.magnitude_px for m in mot],
                             [travel_m / n] * n, [standoff] * n,
                             image_width_px=width)


def _track(n=11, dx=-40.0, standoff=1.0, **kw):
    names = [f"f{i:04d}.jpg" for i in range(n)]
    return M.build_track(frames=names, names=names, times=list(np.arange(n) * 0.1),
                         motions=_motions(n - 1, dx=dx), standoff_m=[standoff] * n,
                         **kw)


# --------------------------------------------------------------------------- #
# scale calibration
# --------------------------------------------------------------------------- #
def test_scale_calibration_recovers_a_known_ground_truth():
    """A pass with known travel and known pixel flow must recover its own scale."""
    # 10 steps x 40 px = 400 px of flow at 1.0 m standoff over 0.80 m of travel
    # -> 2.0 mm/px at 1 m.
    cal = _calibrate()
    assert cal.mm_per_px_at_1m == pytest.approx(2.0, rel=1e-6)
    assert cal.mm_per_px(1.0) == pytest.approx(2.0, rel=1e-6)
    # Ground sampling distance is linear in standoff: twice as far, twice as coarse.
    assert cal.mm_per_px(2.0) == pytest.approx(4.0, rel=1e-6)


def test_scale_calibration_reports_an_implied_hfov_for_sanity_checking():
    cal = _calibrate(width=1280)
    assert 0.0 < cal.implied_hfov_deg < 180.0


def test_scale_calibration_refuses_degenerate_input_rather_than_inventing_a_number():
    """No travel, no flow, or too few pairs has no scale in it. Fail loudly."""
    with pytest.raises(ValueError):
        _calibrate(travel_m=0.0)                     # nothing moved in metres
    with pytest.raises(ValueError):
        _calibrate(dx=0.0)                           # nothing moved in pixels
    with pytest.raises(ValueError):
        _calibrate(n=3)                              # too few pairs to trust


# --------------------------------------------------------------------------- #
# track building
# --------------------------------------------------------------------------- #
def test_track_integrates_motion_into_a_known_distance():
    # 2 mm/px at 1 m standoff, 10 steps x 40 px -> 0.80 m.
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    track = _track(11, scale=cal)
    assert len(track) == 11
    assert track[0].along_m == 0.0
    assert track[-1].along_m == pytest.approx(0.80, abs=1e-3)


def test_unmatched_pairs_are_kept_and_flagged_not_silently_dropped():
    """A gap in matching must stay visible, or coverage silently overstates itself."""
    names = [f"f{i}.jpg" for i in range(5)]
    motions = _motions(4)
    motions[2] = None
    track = M.build_track(names, names, list(np.arange(5) * 0.1), motions,
                          standoff_m=[1.0] * 5)
    assert len(track) == 5
    assert [p.matched for p in track] == [True, True, True, False, True]
    # The unmatched step contributes no motion, so position holds.
    assert track[3].along_m == pytest.approx(track[2].along_m)


def test_drift_grows_with_distance_travelled():
    track = _track(11)
    drifts = [p.drift_m for p in track]
    assert drifts[0] == 0.0
    assert drifts == sorted(drifts), "drift must be monotonically non-decreasing"
    assert drifts[-1] > 0.0


# --------------------------------------------------------------------------- #
# orientation: the map must read forwards, without corrupting the geometry
# --------------------------------------------------------------------------- #
def test_map_is_oriented_so_the_pass_runs_forwards():
    """Whichever way the scene slides, distance from the start reads positive."""
    for dx in (-40.0, +40.0):
        track = _track(11, dx=dx)
        assert track[-1].along_m > 0, f"dx={dx} produced a backwards map"


def test_orientation_is_a_rotation_not_a_mirror():
    """Flipping along without across would silently swap left and right."""
    fwd = _track(11, dx=-40.0)
    rev = M.build_track([f"f{i}.jpg" for i in range(11)], [f"f{i}.jpg" for i in range(11)],
                        list(np.arange(11) * 0.1),
                        [M.FrameMotion(i, 40.0, 10.0, 0.0, 200, 250)
                         for i in range(10)],
                        standoff_m=[1.0] * 11)
    assert rev[-1].orient == -1.0
    assert fwd[-1].orient == 1.0
    # A 180-degree rotation negates both components; a mirror would negate only
    # along. The tell is the across/along RATIO, which a rotation preserves and a
    # mirror inverts — so that is what this asserts.
    #
    # The magnitudes are derived from the scale constant rather than hardcoded.
    # They used to be literals computed against MM_PER_PX_AT_1M = 1.26, so when
    # that constant was corrected to its measured 1.36 this test failed for a
    # reason that had nothing to do with rotation. Ten steps of dx=40 px at 1 m
    # standoff is 400 px of travel.
    expected_along = 10 * 40.0 * M.MM_PER_PX_AT_1M / 1000.0
    assert rev[-1].along_m == pytest.approx(expected_along, rel=1e-3)
    assert rev[-1].across_m == pytest.approx(expected_along * 0.25, rel=1e-3)
    assert rev[-1].across_m / rev[-1].along_m == pytest.approx(0.25, abs=1e-3)


def test_in_frame_offsets_rotate_with_the_path():
    """Regression: rotating the path without rotating in-frame offsets smeared
    every detection by up to half a camera footprint, which silently re-clustered
    sites. A detection at the image centre pins the two conventions together."""
    size = (1280, 720)
    for dx in (-40.0, +40.0):
        track = _track(11, dx=dx)
        centred = {track[5].frame: [{"bbox": [635, 355, 645, 365], "score": 0.9}]}
        det = M.localise_detections(track, centred, size)[0]
        # A detection at image centre is exactly where the camera is, either way.
        assert det.along_m == pytest.approx(track[5].along_m, abs=1e-2)
        assert det.across_m == pytest.approx(track[5].across_m, abs=1e-2)


def test_a_static_object_seen_from_two_positions_lands_in_one_place():
    """The real test of the sign convention: as the camera advances, a fixed
    object slides backwards through the frame, and both effects must cancel."""
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    track = _track(11, dx=-40.0, scale=cal)
    # Frame 5 sees the object 100 px right of centre; by frame 6 the camera has
    # advanced 40 px of flow, so the object has slid to 60 px right of centre.
    dets = {track[5].frame: [{"bbox": [740, 355, 750, 365], "score": 0.9}],
            track[6].frame: [{"bbox": [700, 355, 710, 365], "score": 0.9}]}
    a, b = M.localise_detections(track, dets, (1280, 720))
    assert a.along_m == pytest.approx(b.along_m, abs=1e-2), \
        "the same object seen from two frames must map to one position"


# --------------------------------------------------------------------------- #
# localisation and physical size
# --------------------------------------------------------------------------- #
def test_physical_size_uses_the_standoff_of_the_frame_that_saw_it():
    """The same box in pixels is a bigger object when seen from further away."""
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    near = _track(3, standoff=0.5, scale=cal)
    far = _track(3, standoff=2.0, scale=cal)
    box = {"bbox": [600, 340, 700, 380], "score": 0.9}   # 100 x 40 px
    n = M.localise_detections(near, {near[1].frame: [box]}, (1280, 720))[0]
    f = M.localise_detections(far, {far[1].frame: [box]}, (1280, 720))[0]
    assert n.width_mm == pytest.approx(100 * 2.0 * 0.5)     # 100 mm
    assert f.width_mm == pytest.approx(100 * 2.0 * 2.0)     # 400 mm
    assert f.width_mm == pytest.approx(4 * n.width_mm)


def test_localised_detections_carry_the_drift_of_their_frame():
    track = _track(11)
    dets = {track[10].frame: [{"bbox": [600, 340, 700, 380], "score": 0.5}]}
    d = M.localise_detections(track, dets, (1280, 720))[0]
    assert d.drift_m == track[10].drift_m
    assert "±" in d.describe()


# --------------------------------------------------------------------------- #
# clustering sightings into sites
# --------------------------------------------------------------------------- #
def _det(along, across=0.0, score=0.9):
    return M.LocalisedDetection(frame="f.jpg", frame_index=0, along_m=along,
                                across_m=across, depth_m=1.5, width_mm=50,
                                height_mm=50, score=score, drift_m=0.1)


def test_repeated_sightings_of_one_object_collapse_to_one_site():
    dets = [_det(1.0 + 0.01 * i) for i in range(20)]
    sites = M.cluster_sites(dets, radius_m=0.25)
    assert len(sites) == 1
    assert sites[0].sightings == 20


def test_objects_further_apart_than_the_radius_stay_separate():
    sites = M.cluster_sites([_det(1.0), _det(3.0)], radius_m=0.25)
    assert len(sites) == 2


def test_clustering_is_single_linkage_so_a_chain_is_one_site():
    """Sightings 0.2 m apart in a chain span 1.0 m but are one drifting object."""
    dets = [_det(0.2 * i) for i in range(6)]
    sites = M.cluster_sites(dets, radius_m=0.25)
    assert len(sites) == 1
    assert sites[0].sightings == 6


def test_every_sighting_is_accounted_for_exactly_once():
    dets = [_det(0.1 * i) for i in range(10)] + [_det(5.0), _det(9.0)]
    sites = M.cluster_sites(dets, radius_m=0.25)
    assert sum(s.sightings for s in sites) == len(dets)


def test_sites_are_ranked_with_the_best_evidenced_first():
    dets = [_det(5.0)] + [_det(1.0 + 0.01 * i) for i in range(12)]
    sites = M.cluster_sites(dets, radius_m=0.25)
    assert sites[0].sightings > sites[-1].sightings


def test_confidence_wording_never_claims_more_than_the_evidence():
    """A single sighting must not read as a confirmed find."""
    one = M.cluster_sites([_det(1.0)], radius_m=0.25)[0]
    many = M.cluster_sites([_det(1.0 + 0.01 * i) for i in range(30)], radius_m=0.25)[0]
    assert one.evidence.startswith("weak")
    assert many.evidence.startswith("strong")
    assert one.sightings < many.sightings


def test_clustering_empty_input_is_empty_not_an_error():
    assert M.cluster_sites([], radius_m=0.25) == []


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #
def test_coverage_reports_the_swept_extent():
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    cov = M.coverage(_track(11, scale=cal), image_size=(1280, 720))
    assert cov.along_extent_m == pytest.approx(0.80, abs=1e-2)
    assert cov.swept_area_m2 > 0
    assert cov.gaps == []


def test_coverage_finds_the_band_of_net_that_was_never_photographed():
    """A jump in the track means net went unseen. Saying nothing there would be
    the dangerous failure: an unphotographed hole reads as a clean net."""
    names = [f"f{i}.jpg" for i in range(6)]
    motions = _motions(5)
    motions[2] = M.FrameMotion(2, -4000.0, 0.0, 0.0, 200, 250)   # big jump
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    track = M.build_track(names, names, list(np.arange(6) * 0.1), motions,
                          standoff_m=[1.0] * 6, scale=cal)
    cov = M.coverage(track, image_size=(1280, 720))
    assert cov.gaps, "an 8 m jump between frames must be reported as a gap"
    assert cov.gaps[0]["gap_m"] > 1.0


# --------------------------------------------------------------------------- #
# cross-check against telemetry
# --------------------------------------------------------------------------- #
def test_validation_agrees_when_vision_matches_telemetry():
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    v = M.validate_against_telemetry(_track(11, scale=cal), telemetry_distance_m=0.80)
    assert v["ratio"] == pytest.approx(1.0, abs=0.05)
    assert v["agrees_within_20pct"] is True


def test_validation_flags_disagreement_rather_than_reporting_success():
    cal = M.ScaleCalibration(mm_per_px_at_1m=2.0, reference_standoff_m=1.0,
                             total_pixels=400, total_metres=0.8, frames=10,
                             note="test")
    v = M.validate_against_telemetry(_track(11, scale=cal), telemetry_distance_m=2.0)
    assert v["agrees_within_20pct"] is False


# --------------------------------------------------------------------------- #
# motion estimation on real pixels
# --------------------------------------------------------------------------- #
def test_motion_estimation_recovers_a_known_translation():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (400, 600), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)          # texture ORB can latch onto
    shift = 20
    a, b = base[:, 40:540], base[:, 40 - shift:540 - shift]
    m = M.estimate_motion(a, b)
    assert m is not None, "a textured 20 px shift should match"
    assert m.dx_px == pytest.approx(shift, abs=2.0)
    assert abs(m.dy_px) < 2.0
    assert m.inlier_ratio > 0.5


def test_motion_estimation_returns_none_on_featureless_frames():
    """Blank water gives nothing to match. None is the honest answer."""
    pytest.importorskip("cv2")
    blank = np.full((400, 600), 128, dtype=np.uint8)
    assert M.estimate_motion(blank, blank.copy()) is None

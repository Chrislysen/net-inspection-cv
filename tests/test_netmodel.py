"""Tests for placing a measured strip on a declared cage.

The risk in this module is not a crash, it is a plausible wrong number: a
bearing that drifts the wrong way round the ring, or a coverage figure that
flatters a pass. These check the geometry against hand-computable answers.
"""
from __future__ import annotations

import math

import pytest

from netinspect import netmodel as N


def _geom(**kw):
    base = dict(circumference_m=160.0, cylinder_depth_m=15.0, cone_depth_m=10.0,
                barge_bearing_deg=0.0, start_bearing_deg=0.0, clockwise=True)
    base.update(kw)
    return N.PenGeometry(**base)


# --------------------------------------------------------------------------- #
# cage dimensions
# --------------------------------------------------------------------------- #
def test_radius_follows_from_circumference():
    g = _geom(circumference_m=2 * math.pi * 25.0)
    assert g.radius_m == pytest.approx(25.0)


def test_net_area_is_wall_plus_cone():
    g = _geom()
    wall = 160.0 * 15.0
    cone = math.pi * g.radius_m * math.hypot(g.radius_m, 10.0)
    assert g.net_area_m2 == pytest.approx(wall + cone)
    assert g.net_area_m2 > wall, "the cone is netting too and must be counted"


def test_a_cage_with_no_cone_still_has_a_flat_bottom_net():
    """Zero cone depth is not zero bottom: the taper collapses to a flat disc,
    which is still netting a fish can leave through."""
    g = _geom(cone_depth_m=0.0)
    assert g.cone_slant_m == pytest.approx(g.radius_m)
    assert g.net_area_m2 == pytest.approx(160.0 * 15.0 + math.pi * g.radius_m ** 2)


def test_impossible_cages_are_rejected():
    with pytest.raises(ValueError):
        _geom(circumference_m=0.0)
    with pytest.raises(ValueError):
        _geom(cylinder_depth_m=-1.0)
    with pytest.raises(ValueError):
        _geom(cylinder_depth_m=0.0, cone_depth_m=0.0)


# --------------------------------------------------------------------------- #
# bearings around the ring
# --------------------------------------------------------------------------- #
def test_a_quarter_of_the_circumference_is_ninety_degrees():
    g = _geom()
    assert N.bearing_at(40.0, g) == pytest.approx(90.0)
    assert N.bearing_at(80.0, g) == pytest.approx(180.0)


def test_travelling_anticlockwise_turns_the_other_way():
    assert N.bearing_at(40.0, _geom(clockwise=False)) == pytest.approx(270.0)


def test_bearings_wrap_rather_than_running_past_360():
    assert N.bearing_at(160.0, _geom()) == pytest.approx(0.0)
    assert 0.0 <= N.bearing_at(500.0, _geom()) < 360.0


def test_the_start_bearing_offsets_the_whole_pass():
    assert N.bearing_at(40.0, _geom(start_bearing_deg=90.0)) == pytest.approx(180.0)


# --------------------------------------------------------------------------- #
# the net narrows in the cone
# --------------------------------------------------------------------------- #
def test_radius_is_constant_down_the_wall_then_tapers():
    g = _geom()
    assert N.radius_at_depth(0.0, g) == pytest.approx(g.radius_m)
    assert N.radius_at_depth(15.0, g) == pytest.approx(g.radius_m)
    assert N.radius_at_depth(20.0, g) == pytest.approx(g.radius_m / 2)
    assert N.radius_at_depth(25.0, g) == pytest.approx(0.0, abs=1e-9)


def test_below_the_cage_does_not_go_negative():
    assert N.radius_at_depth(999.0, _geom()) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# placing a position
# --------------------------------------------------------------------------- #
def test_a_position_lands_on_the_wall_at_the_right_compass_point():
    g = _geom()
    p = N.place_on_pen(40.0, 2.0, g)
    assert p.bearing_deg == pytest.approx(90.0)
    assert p.x_m == pytest.approx(g.radius_m, abs=1e-3)     # due east
    assert p.y_m == pytest.approx(0.0, abs=1e-3)
    assert p.z_m == pytest.approx(-2.0)
    assert p.section == "wall"


def test_a_deep_position_is_reported_as_being_on_the_cone():
    p = N.place_on_pen(0.0, 20.0, _geom())
    assert p.section == "cone"


def test_distance_from_the_barge_takes_the_shorter_way_round():
    g = _geom(barge_bearing_deg=0.0)
    # Three quarters clockwise is a quarter anticlockwise — 40 m, not 120 m.
    p = N.place_on_pen(120.0, 1.0, g)
    assert p.arc_from_barge_m == pytest.approx(40.0, abs=0.1)
    assert p.side == "anticlockwise"


def test_the_near_side_of_the_ring_reads_clockwise():
    p = N.place_on_pen(20.0, 1.0, _geom())
    assert p.side == "clockwise"
    assert p.arc_from_barge_m == pytest.approx(20.0, abs=0.1)


def test_a_position_at_the_barge_says_so_instead_of_a_bearing():
    p = N.place_on_pen(0.0, 1.0, _geom(barge_bearing_deg=0.0))
    assert p.side == "at the barge"
    assert "feed barge" in p.describe()


def test_the_description_is_something_a_person_could_follow():
    text = N.place_on_pen(40.0, 1.7, _geom()).describe()
    assert "feed barge" in text and "deep" in text and "wall" in text


# --------------------------------------------------------------------------- #
# the barge landmark
# --------------------------------------------------------------------------- #
def test_the_barge_sits_outside_the_pen_wall():
    g = _geom(barge_bearing_deg=0.0, barge_offset_m=18.0)
    b = N.barge_anchor(g)
    assert b["distance_from_centre_m"] == pytest.approx(g.radius_m + 18.0, abs=1e-2)
    assert b["y_m"] == pytest.approx(g.radius_m + 18.0, abs=1e-2)   # due north
    assert b["x_m"] == pytest.approx(0.0, abs=1e-2)


def test_moving_the_barge_moves_the_landmark():
    b = N.barge_anchor(_geom(barge_bearing_deg=90.0))
    assert b["x_m"] > 0 and b["y_m"] == pytest.approx(0.0, abs=1e-2)


# --------------------------------------------------------------------------- #
# coverage against a whole cage
# --------------------------------------------------------------------------- #
def test_coverage_against_a_real_cage_is_a_fraction_of_a_percent():
    """The number that stops a 5.5 m pass sounding like an inspected net."""
    cov = N.coverage_of_net(6.3, 5.5, _geom())
    assert cov["area_percent"] < 1.0
    assert cov["ring_percent"] == pytest.approx(100 * 5.5 / 160.0, abs=0.01)


def test_coverage_reports_how_many_passes_the_ring_would_take():
    cov = N.coverage_of_net(6.3, 5.5, _geom())
    assert cov["passes_to_cover_ring"] == math.ceil(160.0 / 5.5)


def test_a_full_circuit_is_one_pass_of_the_ring():
    cov = N.coverage_of_net(100.0, 160.0, _geom())
    assert cov["ring_percent"] == pytest.approx(100.0)
    assert cov["passes_to_cover_ring"] == 1


# --------------------------------------------------------------------------- #
# projecting mapped sites
# --------------------------------------------------------------------------- #
def _site(site_id=1, along=40.0, depth=1.7, sightings=12):
    return {"site_id": site_id, "along_m": along, "across_m": 0.0, "depth_m": depth,
            "sightings": sightings, "evidence": "strong", "median_width_mm": 50,
            "median_height_mm": 50, "max_score": 0.9}


def test_sites_keep_their_measured_evidence_when_placed():
    out = N.project_sites([_site()], _geom())[0]
    assert out["sightings"] == 12
    assert out["measured"]["along_m"] == 40.0
    assert out["placed"]["bearing_deg"] == pytest.approx(90.0)


def test_a_site_without_a_depth_does_not_crash_the_placement():
    s = _site(); s["depth_m"] = None
    out = N.project_sites([s], _geom())[0]
    assert out["placed"]["z_m"] == 0.0


def test_scene_separates_what_was_measured_from_what_was_declared():
    """The whole point: a viewer must be able to tell the shell from the data."""
    scene = N.build_scene(
        sites=[_site()],
        track=[{"along_m": 0.0, "depth_m": 1.6, "standoff_m": 0.6, "mm_per_px": 0.83},
               {"along_m": 5.5, "depth_m": 1.7, "standoff_m": 0.8, "mm_per_px": 1.1}],
        coverage={"along_extent_m": 5.5, "swept_area_m2": 6.3, "gaps": []},
        geom=_geom())
    assert scene["pen"]["declared"] is True
    assert "circumference" in " ".join(scene["provenance"]["declared"])
    assert scene["provenance"]["measured"], "measured items must be listed too"
    assert scene["coverage"]["area_percent"] < 1.0
    assert len(scene["band"]) == 2
    assert scene["band"][1]["bearing_deg"] > scene["band"][0]["bearing_deg"]
    assert scene["barge"]["label"] == "feed barge"

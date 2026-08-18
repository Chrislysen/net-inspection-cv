"""Tests for ROV telemetry extraction and the frame-to-telemetry join.

No ROS bags and no ``rosbags`` install: the field extractors take deserialised
message objects, so they are tested against stand-ins that carry the same
attributes. That keeps the suite runnable in CI, and it isolates the part most
likely to be wrong — the per-message field mapping — from the bag reader.

The sensor-suite drift covered here is real: the two SOLAQUA recording days use
different DVL hardware reporting different units, and silently merging them
would manufacture a step change at the day boundary.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from netinspect.telemetry import (
    SPECS_BY_NAME,
    STREAM_SPECS,
    THRUST_INVALID,
    _extract_attitude,
    _extract_depth_temp,
    _extract_dvl_a50,
    _extract_dvl_nucleus,
    _extract_guidance,
    _extract_imu_counts,
    _extract_imu_si,
    _extract_net_plane,
    _extract_setpoint,
    _extract_thrust,
    _header_time,
    clip_id,
    describe_streams,
    merge_on_times,
    summarise,
)


# --------------------------------------------------------------------------- #
# Clip identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path,expected", [
    ("data/raw/solaqua/2024-08-22_14-47-39_data.bag", "2024-08-22_14-47-39"),
    ("data/raw/solaqua/2024-08-22_14-47-39_video.bag", "2024-08-22_14-47-39"),
    ("/abs/path/2024-08-20_15-18-27_data.bag", "2024-08-20_15-18-27"),
])
def test_video_and_data_bags_share_a_clip_id(path, expected):
    """The join depends on both bags of a run mapping to the same key."""
    assert clip_id(path) == expected


def test_clip_id_passes_through_unrecognised_names():
    assert clip_id("something_else.bag") == "something_else"


# --------------------------------------------------------------------------- #
# Net-plane: the signal the whole analysis rests on
# --------------------------------------------------------------------------- #
def test_net_plane_extracts_standoff_and_lock():
    msg = NS(NetDistance=0.53, NetHeading=-0.064, NetPitch=-0.025, NetLock=1.0,
             NetVelocity_u=-0.031, NetVelocity_v=-0.075, NetVelocity_w=-7e-05,
             Altitude=0.465, NormalDVL=[-1.0, 0.036, 0.215])
    out = _extract_net_plane(msg)
    assert out["net_distance"] == pytest.approx(0.53)
    assert out["net_lock"] == 1.0
    assert out["net_normal_x"] == pytest.approx(-1.0)
    assert out["net_normal_z"] == pytest.approx(0.215)


def test_net_plane_survives_a_missing_normal_vector():
    out = _extract_net_plane(NS(NetDistance=0.6, NetLock=0.0))
    assert out["net_distance"] == 0.6
    assert "net_normal_x" not in out


def test_missing_fields_become_nan_not_zero():
    """A missing reading must not look like a measured zero."""
    out = _extract_net_plane(NS())
    assert out["net_distance"] != out["net_distance"]        # NaN


# --------------------------------------------------------------------------- #
# Sensor-suite drift between the two recording days
# --------------------------------------------------------------------------- #
def test_both_dvl_suites_map_onto_the_same_columns():
    """The two days use different DVL hardware; columns must line up."""
    a50 = _extract_dvl_a50(NS(velocity=NS(x=-0.007, y=0.024, z=0.044),
                              altitude=0.647, fom=0.0125, velocity_valid=True))
    nucleus = _extract_dvl_nucleus(NS(dvl_velocity_xyz=NS(x=-0.003, y=-0.184, z=0.023),
                                      beam_distance=[0.453, 0.475, 0.179],
                                      dvl_fom_xyz=NS(x=0.015, y=0.026, z=0.005),
                                      data_valid=True))
    assert set(a50) == set(nucleus)
    assert a50["valid"] == 1.0 and nucleus["valid"] == 1.0


def test_nucleus_altitude_averages_only_valid_beams():
    out = _extract_dvl_nucleus(NS(dvl_velocity_xyz=NS(x=0, y=0, z=0),
                                  beam_distance=[0.4, 0.6, 0.0, -1.0],
                                  data_valid=True))
    assert out["altitude"] == pytest.approx(0.5)


def test_nucleus_altitude_is_nan_when_no_beam_is_valid():
    out = _extract_dvl_nucleus(NS(dvl_velocity_xyz=NS(x=0, y=0, z=0),
                                  beam_distance=[0.0, -1.0], data_valid=False))
    assert out["altitude"] != out["altitude"]
    assert out["valid"] == 0.0


def test_the_two_imus_are_separate_streams_because_units_differ():
    """A50 reports counts (~milli-g); Nortek reports SI. Merging would lie."""
    assert "imu_counts" in SPECS_BY_NAME and "imu_si" in SPECS_BY_NAME
    counts = _extract_imu_counts(NS(acc_x=-7.0, acc_y=38.0, acc_z=-1012.0,
                                    gyro_x=-7.0, gyro_y=-98.0, gyro_z=-17.0,
                                    mag_x=0.0, mag_y=0.0, mag_z=0.0))
    si = _extract_imu_si(NS(accelerometer=NS(x=0.026, y=9.729, z=-0.699),
                            gyroscope=NS(x=-0.004, y=-0.008, z=-0.003)))
    assert abs(counts["acc_z"]) > 100      # counts
    assert abs(si["acc_y"]) < 20           # m/s^2
    assert SPECS_BY_NAME["imu_counts"].name != SPECS_BY_NAME["imu_si"].name


# --------------------------------------------------------------------------- #
# Remaining extractors
# --------------------------------------------------------------------------- #
def test_depth_temperature_extraction():
    out = _extract_depth_temp(NS(pressure=1194.7, depth=1.855, temperature=15.54))
    assert out["depth"] == pytest.approx(1.855)
    assert out["temperature"] == pytest.approx(15.54)


def test_setpoint_captures_operator_intent():
    """Commanded values are what make an inspection auditable."""
    out = _extract_setpoint(NS(d_depth=0.0, d_net_distance=0.6,
                               d_net_velocity_horizontal=-0.1,
                               d_net_velocity_vertical=0.0, offset_net_heading=0.0))
    assert out["d_net_distance"] == pytest.approx(0.6)
    assert out["d_net_velocity_horizontal"] == pytest.approx(-0.1)


def test_guidance_reports_error_and_desired():
    out = _extract_guidance(NS(error_net_distance=-0.06, desired_net_distance=0.6,
                               error_surge=-0.063, desired_surge=-0.06))
    assert out["error_net_distance"] == pytest.approx(-0.06)
    assert out["desired_net_distance"] == pytest.approx(0.6)


def test_attitude_extraction():
    out = _extract_attitude(NS(roll=-0.004, pitch=-0.036, yaw=1.673,
                               rollspeed=0.017, pitchspeed=0.004, yawspeed=0.037))
    assert out["yaw"] == pytest.approx(1.673)
    assert out["yawspeed"] == pytest.approx(0.037)


def test_thrust_maps_the_absent_channel_sentinel_to_nan():
    """65535 means 'channel not present' — averaging it in would be nonsense."""
    out = _extract_thrust(NS(data=[1500.0, 1539.0, THRUST_INVALID, 1461.0]))
    assert out["thrust_2"] != out["thrust_2"]
    assert out["thrust_effort"] == pytest.approx((0 + 39 + 39) / 3)


def test_thrust_effort_is_nan_when_every_channel_is_absent():
    out = _extract_thrust(NS(data=[THRUST_INVALID] * 4))
    assert out["thrust_effort"] != out["thrust_effort"]


def test_thrust_handles_a_missing_data_field():
    assert _extract_thrust(NS())["thrust_effort"] != _extract_thrust(NS())["thrust_effort"]


# --------------------------------------------------------------------------- #
# Header time
# --------------------------------------------------------------------------- #
def test_header_time_combines_sec_and_nanosec():
    msg = NS(header=NS(stamp=NS(sec=1724330866, nanosec=274127006)))
    assert _header_time(msg) == pytest.approx(1724330866.274127, abs=1e-5)


def test_header_time_is_nan_without_a_header():
    """/commanded_thrust has no header at all — that must not raise."""
    assert _header_time(NS()) != _header_time(NS())


# --------------------------------------------------------------------------- #
# Stream specification
# --------------------------------------------------------------------------- #
def test_stream_names_are_unique():
    names = [s.name for s in STREAM_SPECS]
    assert len(names) == len(set(names))


def test_every_stream_declares_topics_and_a_description():
    for s in STREAM_SPECS:
        assert s.topics and all(t.startswith("/") for t in s.topics), s.name
        assert s.description.strip(), s.name


def test_dvl_lists_both_suites_with_the_shared_topic_first():
    """Preference order matters: the A50 topic is common to both days."""
    topics = SPECS_BY_NAME["dvl"].topics
    assert topics[0] == "/sensor/dvl_velocity"
    assert "/nucleus1000dvl/bottomtrack" in topics


def test_describe_streams_is_serialisable():
    described = describe_streams()
    assert len(described) == len(STREAM_SPECS)
    assert all(isinstance(d["topics"], list) for d in described)


# --------------------------------------------------------------------------- #
# As-of join — the piece the frame analysis depends on
# --------------------------------------------------------------------------- #
def _frame(pd, times, values):
    return pd.DataFrame({"t": times, "net_distance": values})


def test_merge_matches_nearest_sample_within_tolerance():
    pd = pytest.importorskip("pandas")
    streams = {"net_plane": _frame(pd, [100.0, 101.0, 102.0], [0.6, 0.7, 0.8])}
    out = merge_on_times(streams, [100.1, 101.9], tolerance_s=0.5)
    assert out["net_plane_net_distance"].tolist() == [0.6, 0.8]


def test_merge_yields_nan_beyond_tolerance_rather_than_stale_values():
    """A frame with no nearby sample must be visibly unmatched, not imputed."""
    pd = pytest.importorskip("pandas")
    streams = {"net_plane": _frame(pd, [100.0], [0.6])}
    out = merge_on_times(streams, [200.0], tolerance_s=0.5)
    assert out["net_plane_net_distance"].isna().all()


def test_merge_prefixes_columns_so_provenance_survives():
    pd = pytest.importorskip("pandas")
    streams = {"net_plane": _frame(pd, [100.0], [0.6]),
               "dvl": pd.DataFrame({"t": [100.0], "altitude": [0.65]})}
    out = merge_on_times(streams, [100.0])
    assert "net_plane_net_distance" in out.columns
    assert "dvl_altitude" in out.columns


def test_merge_skips_empty_streams():
    pd = pytest.importorskip("pandas")
    streams = {"net_plane": _frame(pd, [100.0], [0.6]), "dvl": pd.DataFrame()}
    out = merge_on_times(streams, [100.0])
    assert "net_plane_net_distance" in out.columns


def test_merge_can_restrict_to_named_streams():
    pd = pytest.importorskip("pandas")
    streams = {"net_plane": _frame(pd, [100.0], [0.6]),
               "dvl": pd.DataFrame({"t": [100.0], "altitude": [0.65]})}
    out = merge_on_times(streams, [100.0], include=["net_plane"])
    assert "dvl_altitude" not in out.columns


def test_merge_returns_one_row_per_query_time():
    pd = pytest.importorskip("pandas")
    streams = {"net_plane": _frame(pd, [100.0, 101.0], [0.6, 0.7])}
    assert len(merge_on_times(streams, [100.0, 100.5, 101.0])) == 3


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_summarise_reports_rate_and_source_topic():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"t": [0.0, 1.0, 2.0], "net_distance": [0.6, 0.6, 0.6],
                       "source_topic": ["/navigation/plane_approximation"] * 3})
    out = summarise({"net_plane": df})
    assert out["net_plane"]["rows"] == 3
    assert out["net_plane"]["duration_s"] == pytest.approx(2.0)
    assert out["net_plane"]["rate_hz"] == pytest.approx(1.5)
    assert out["net_plane"]["source_topic"] == "/navigation/plane_approximation"


def test_summarise_skips_empty_streams():
    pd = pytest.importorskip("pandas")
    assert summarise({"dvl": pd.DataFrame()}) == {}

"""Tests for recovering frame timestamps and joining frames to telemetry.

The join is what lets a model output be conditioned on the conditions of
capture, so its failure modes matter more than its happy path. Two in
particular are covered here: a frame whose timestamp cannot be recovered must
stay visible rather than vanish, and a frame with no telemetry inside the
tolerance must read as unmatched rather than inherit a stale value.

No bags required — the frame index is a plain JSON structure, so it is built
directly.
"""
from __future__ import annotations

import pytest

from netinspect.frame_sync import (
    coverage_report,
    frame_records,
    frame_time,
    join_frames,
    parse_frame_name,
)


# --------------------------------------------------------------------------- #
# Filename parsing — the bridge from file back to message index
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,clip,index", [
    ("2024-08-22_14-47-39_video_000123.jpg", "2024-08-22_14-47-39", 123),
    ("2024-08-22_14-47-39_video_000000.jpg", "2024-08-22_14-47-39", 0),
    ("2024-08-20_15-18-27_sonar_000042.png", "2024-08-20_15-18-27", 42),
])
def test_parses_clip_and_message_index(name, clip, index):
    assert parse_frame_name(name) == (clip, index)


@pytest.mark.parametrize("name", [
    "random.jpg", "no_index_here.jpg", "2024-08-22_14-47-39.jpg", "video_abc.jpg",
])
def test_unrecognised_names_return_none_rather_than_raising(name):
    """A stray file in the frame directory should be skipped, not fatal."""
    assert parse_frame_name(name) is None


def test_parsing_accepts_a_full_path():
    assert parse_frame_name("/a/b/2024-08-22_14-06-43_video_000007.jpg") \
        == ("2024-08-22_14-06-43", 7)


# --------------------------------------------------------------------------- #
# Index lookup
# --------------------------------------------------------------------------- #
def _index(clip="c", times=(100.0, 100.5, 101.0)):
    return {"clip": clip, "topic": "/image/compressed_image/data",
            "count": len(times), "times": list(times)}


def test_frame_time_reads_the_index():
    assert frame_time(_index(), 1) == 100.5


@pytest.mark.parametrize("idx", [-1, 3, 999])
def test_out_of_range_index_is_nan_not_an_exception(idx):
    """Frame files can outlive a shorter index; that must degrade gracefully."""
    t = frame_time(_index(), idx)
    assert t != t


def test_frame_time_handles_an_index_with_no_times():
    t = frame_time({"clip": "c", "times": []}, 0)
    assert t != t


# --------------------------------------------------------------------------- #
# Building records from a directory
# --------------------------------------------------------------------------- #
def _write_frames(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"\xff\xd8\xff\xdb")   # minimal JPEG marker
    return tmp_path


def test_records_carry_clip_index_and_time(tmp_path):
    d = _write_frames(tmp_path, ["c_video_000000.jpg", "c_video_000001.jpg"])
    recs = frame_records(d, _index(times=(100.0, 100.5)))
    assert [r["msg_index"] for r in recs] == [0, 1]
    assert [r["t"] for r in recs] == [100.0, 100.5]
    assert all(r["clip"] == "c" for r in recs)


def test_records_accept_a_clip_to_index_mapping(tmp_path):
    d = _write_frames(tmp_path, ["a_video_000000.jpg", "b_video_000000.jpg"])
    recs = frame_records(d, {"a": _index("a", (10.0,)), "b": _index("b", (20.0,))})
    assert sorted(r["t"] for r in recs) == [10.0, 20.0]


def test_frames_from_an_unknown_clip_are_kept_with_nan_time(tmp_path):
    """Dropping them would silently shrink the denominator."""
    d = _write_frames(tmp_path, ["known_video_000000.jpg", "other_video_000000.jpg"])
    recs = frame_records(d, {"known": _index("known", (10.0,))})
    assert len(recs) == 2
    untimed = [r for r in recs if r["t"] != r["t"]]
    assert len(untimed) == 1 and untimed[0]["clip"] == "other"


def test_non_frame_files_are_ignored(tmp_path):
    d = _write_frames(tmp_path, ["c_video_000000.jpg", "notes.jpg"])
    assert len(frame_records(d, _index(times=(1.0,)))) == 1


def test_empty_directory_yields_no_records(tmp_path):
    assert frame_records(tmp_path, _index()) == []


# --------------------------------------------------------------------------- #
# The join
# --------------------------------------------------------------------------- #
def _telemetry(pd, times, values):
    return {"net_plane": pd.DataFrame({"t": list(times),
                                       "net_distance": list(values)})}


def test_join_attaches_telemetry_to_each_frame(tmp_path):
    pd = pytest.importorskip("pandas")
    d = _write_frames(tmp_path, ["c_video_000000.jpg", "c_video_000001.jpg"])
    out = join_frames(d, _telemetry(pd, [100.0, 100.5], [0.60, 0.72]),
                      _index(times=(100.0, 100.5)))
    assert len(out) == 2
    assert out["net_plane_net_distance"].tolist() == [0.60, 0.72]


def test_join_leaves_far_frames_unmatched(tmp_path):
    pd = pytest.importorskip("pandas")
    d = _write_frames(tmp_path, ["c_video_000000.jpg"])
    out = join_frames(d, _telemetry(pd, [999.0], [0.6]), _index(times=(100.0,)),
                      tolerance_s=0.5)
    assert out["net_plane_net_distance"].isna().all()


def test_join_keeps_untimeable_frames_in_the_output(tmp_path):
    pd = pytest.importorskip("pandas")
    d = _write_frames(tmp_path, ["c_video_000000.jpg", "c_video_000009.jpg"])
    out = join_frames(d, _telemetry(pd, [100.0], [0.6]), _index(times=(100.0,)))
    assert len(out) == 2                       # index has one entry, dir has two
    assert out["t"].isna().sum() == 1


def test_join_of_an_empty_directory_returns_empty(tmp_path):
    pd = pytest.importorskip("pandas")
    out = join_frames(tmp_path, _telemetry(pd, [100.0], [0.6]), _index())
    assert len(out) == 0


def test_join_output_is_ordered_by_clip_then_index(tmp_path):
    pd = pytest.importorskip("pandas")
    d = _write_frames(tmp_path, ["c_video_000002.jpg", "c_video_000000.jpg",
                                 "c_video_000001.jpg"])
    out = join_frames(d, _telemetry(pd, [100.0, 100.5, 101.0], [0.1, 0.2, 0.3]),
                      _index(times=(100.0, 100.5, 101.0)))
    assert out["msg_index"].tolist() == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Coverage — print this before trusting any conditional statistic
# --------------------------------------------------------------------------- #
def test_coverage_reports_timestamped_and_matched_shares(tmp_path):
    pd = pytest.importorskip("pandas")
    d = _write_frames(tmp_path, ["c_video_000000.jpg", "c_video_000001.jpg"])
    out = join_frames(d, _telemetry(pd, [100.0], [0.6]),
                      _index(times=(100.0, 500.0)), tolerance_s=0.5)
    cov = coverage_report(out)
    assert cov["frames"] == 2
    assert cov["timestamped"] == 2
    assert cov["matched"] == 1
    assert cov["matched_pct"] == 50.0


def test_coverage_of_an_empty_join():
    pd = pytest.importorskip("pandas")
    assert coverage_report(pd.DataFrame({"t": []}))["frames"] == 0


def test_coverage_handles_a_missing_match_key(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"t": [1.0, 2.0]})
    assert coverage_report(df, key="not_a_column")["matched"] == 0

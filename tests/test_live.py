"""Tests for real-time inspection: source handling, confirmation, and the session.

The threading is the part worth testing hardest. A live pipeline that quietly
builds a backlog still *looks* fine — smooth video, plausible boxes — while
showing the operator a progressively older world. So the tests below assert the
drop behaviour explicitly rather than just that frames come out.

A small video file is generated on the fly so the capture path runs for real
rather than against a mock; everything else uses a stub detector so the tests
stay fast and independent of model weights.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from netinspect.live import (
    LiveFrame,
    LiveInspector,
    LiveSession,
    LiveSource,
    describe_source,
    parse_source,
)
from netinspect.utils import BBox

cv2 = pytest.importorskip("cv2", reason="live capture needs OpenCV")


# --------------------------------------------------------------------------- #
# Source interpretation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("0", 0), ("1", 1), (0, 0),
    ("rtsp://cam/stream", "rtsp://cam/stream"),
    ("video.mp4", "video.mp4"),
])
def test_source_strings_are_interpreted_the_way_an_operator_writes_them(raw, expected):
    """'0' means webcam zero, not a file named 0."""
    assert parse_source(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("0", "USB camera 0"),
    ("rtsp://cam/s", "RTSP stream"),
    ("http://cam/mjpg", "HTTP stream"),
    ("clip.mp4", "video file"),
])
def test_source_kinds_are_described_for_the_ui(raw, expected):
    assert describe_source(raw) == expected


def test_opening_a_missing_source_raises_an_actionable_error(tmp_path):
    src = LiveSource(str(tmp_path / "nope.mp4"))
    with pytest.raises(RuntimeError, match="Could not open"):
        src.open()


# --------------------------------------------------------------------------- #
# A real (tiny) video file
# --------------------------------------------------------------------------- #
@pytest.fixture
def video(tmp_path):
    """Twelve frames of moving colour, written as a real file."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():                        # codec unavailable on this box
        pytest.skip("no mp4v encoder available")
    for i in range(12):
        frame = np.full((48, 64, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("video file was not produced")
    return path


def test_reads_frames_as_rgb(video):
    with LiveSource(str(video)) as src:
        frame = src.read()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_end_of_file_is_not_an_error(video):
    with LiveSource(str(video), loop_files=False) as src:
        seen = 0
        while src.read() is not None and seen < 100:
            seen += 1
    assert 0 < seen < 100                            # terminated, did not hang


def test_a_file_can_loop_so_a_clip_stands_in_for_a_camera(video):
    with LiveSource(str(video), loop_files=True) as src:
        frames = [src.read() for _ in range(30)]
    assert all(f is not None for f in frames)        # more reads than the clip has


# --------------------------------------------------------------------------- #
# Inference over frames
# --------------------------------------------------------------------------- #
class _StubResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.heatmap = None
        self.elapsed_ms = 1.0
        self.meta = {}


class _StubInspector:
    """Returns the same box every frame, so temporal confirmation can be observed."""
    def __init__(self, boxes_per_frame=1, delay_s=0.0):
        self.boxes_per_frame = boxes_per_frame
        self.delay_s = delay_s
        self.calls = 0

    def predict(self, image, method="yolo", conf=0.25):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        boxes = [BBox(10, 10, 30, 30, 0, "damage", 0.9)
                 for _ in range(self.boxes_per_frame)]
        return _StubResult(boxes)


def _rgb():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_process_returns_a_populated_frame():
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    out = live.process(_rgb(), index=7)
    assert isinstance(out, LiveFrame)
    assert out.index == 7
    assert out.raw_detections and out.confirmed
    assert out.latency_ms >= 0


def test_a_detection_must_persist_before_it_is_confirmed():
    """One-frame flickers must not reach an operator."""
    live = LiveInspector(_StubInspector(), min_hits=3, draw=False)
    confirmed = [len(live.process(_rgb(), i).confirmed) for i in range(4)]
    assert confirmed[0] == 0 and confirmed[1] == 0
    assert confirmed[2] >= 1


def test_no_detections_means_no_confirmations():
    live = LiveInspector(_StubInspector(boxes_per_frame=0), min_hits=1, draw=False)
    assert live.process(_rgb()).confirmed == []


def test_drawing_can_be_disabled_for_headless_use():
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    out = live.process(_rgb())
    assert out.image.shape == (48, 64, 3)


def test_frame_serialises_for_the_api():
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    d = live.process(_rgb(), index=3).to_dict()
    assert d["index"] == 3 and d["confirmed"] >= 1
    assert isinstance(d["detections"], list)


def test_run_stops_at_end_of_source(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    with LiveSource(str(video), loop_files=False) as src:
        frames = list(live.run(src))
    assert 0 < len(frames) <= 12


def test_run_can_skip_frames(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    with LiveSource(str(video), loop_files=False) as src:
        frames = list(live.run(src, every_n=3))
    assert 0 < len(frames) <= 5


def test_run_respects_max_frames(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    with LiveSource(str(video), loop_files=True) as src:
        assert len(list(live.run(src, max_frames=4))) == 4


# --------------------------------------------------------------------------- #
# The threaded session
# --------------------------------------------------------------------------- #
def _wait_for(predicate, timeout=8.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_session_produces_frames_and_stops_cleanly(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    session = LiveSession(str(video), live, loop_files=True)
    session.start()
    try:
        assert _wait_for(lambda: session.frames_inferred > 2)
        assert session.latest() is not None
        assert session.running
    finally:
        session.stop()
    assert not session.running


def test_session_drops_frames_rather_than_building_a_backlog(video):
    """The property that makes it real-time instead of delayed playback."""
    slow = LiveInspector(_StubInspector(delay_s=0.05), min_hits=1, draw=False)
    session = LiveSession(str(video), slow, loop_files=True)
    session.start()
    try:
        assert _wait_for(lambda: session.frames_dropped > 0, timeout=8.0)
        assert session.frames_captured > session.frames_inferred
    finally:
        session.stop()


def test_status_reports_what_an_operator_needs(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    session = LiveSession(str(video), live, loop_files=True)
    session.start()
    try:
        _wait_for(lambda: session.frames_inferred > 1)
        s = session.status()
        assert s["running"] is True
        assert s["frames_inferred"] >= 1
        assert s["inference_fps"] >= 0
        assert "synthetic" in s["disclaimer"].lower()
    finally:
        session.stop()
    assert session.status()["running"] is False


def test_a_persisting_defect_is_one_event_not_one_per_frame(video):
    """The stub reports a box on every frame; that must be a single event."""
    live = LiveInspector(_StubInspector(), min_hits=2, draw=False)
    session = LiveSession(str(video), live, loop_files=True)
    session.start()
    try:
        assert _wait_for(lambda: session.frames_inferred > 8)
        assert len(session.events) < session.frames_inferred
        assert len(session.events) >= 1
    finally:
        session.stop()


def test_starting_twice_is_refused(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    session = LiveSession(str(video), live, loop_files=True)
    session.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            session.start()
    finally:
        session.stop()


def test_starting_a_bad_source_fails_immediately_not_in_a_thread(tmp_path):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    session = LiveSession(str(tmp_path / "missing.mp4"), live)
    with pytest.raises(RuntimeError):
        session.start()
    assert not session.running


def test_status_before_start_is_safe(video):
    live = LiveInspector(_StubInspector(), min_hits=1, draw=False)
    s = LiveSession(str(video), live).status()
    assert s["running"] is False and s["frames_inferred"] == 0

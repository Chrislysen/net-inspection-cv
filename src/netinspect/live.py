"""Real-time inspection of a live camera, ROV feed, or video file.

Everything needed to point the inference facade at a moving picture and keep up
with it. The desktop runner (``scripts/live_inspect.py``) and the browser
console (``scripts/serve.py``) both drive this module, so live behaviour is
identical whichever way it is watched — a difference between the two would be a
bug that only shows up in a demo.

Three things make it real-time rather than merely "runs on video":

**Capture and inference are decoupled.** A camera delivers frames on its own
schedule; a model does not. :class:`LiveSession` captures on one thread and
infers on another, and the inference thread always takes the *newest* frame,
dropping whatever queued behind it. The alternative — a single loop — silently
turns into playback of an ever-growing backlog, which looks fine on a video file
and falls apart on a camera.

**Detections are confirmed over time.** A defect that appears for one frame and
never again is almost always a flicker. :mod:`netinspect.temporal` tracks
detections across frames so an operator gets one event per *persisting* defect
rather than one per frame.

**Unfamiliar frames are flagged, not silently scored.** A new site, camera or
water condition is outside anything these models were characterised on, and the
OOD gate says so instead of quietly producing confident boxes.

Honesty
-------
The shipped models learned **synthetic** damage. On a live feed from a real net,
treat every detection as *human-review triage*, not a verified alarm — and
expect the OOD gate to flag a new domain, because it is one. Run in shadow mode
and fine-tune on real labels before anything alerts.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from .mapping import MM_PER_PX_AT_1M
from .utils import BBox, get_logger, optional_import, require

LOGGER = get_logger()

# A live feed is watched, not archived: hold a short event history and drop the rest.
MAX_EVENTS = 200
RECONNECT_DELAY_S = 2.0
MAX_RECONNECT_ATTEMPTS = 5


def parse_source(source: str | int) -> int | str:
    """Interpret a source string the way an operator would write it.

    ``"0"`` is webcam index 0, not a file called "0". Anything with a scheme is
    a network stream; everything else is a path.
    """
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s


def describe_source(source: str | int) -> str:
    """Human-readable kind, for the UI and the logs."""
    parsed = parse_source(source)
    if isinstance(parsed, int):
        return f"USB camera {parsed}"
    lower = str(parsed).lower()
    for scheme, label in (("rtsp://", "RTSP stream"), ("rtmp://", "RTMP stream"),
                          ("http://", "HTTP stream"), ("https://", "HTTP stream")):
        if lower.startswith(scheme):
            return label
    return "video file"


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
class LiveSource:
    """A video source that reopens itself when a network stream drops.

    Wraps ``cv2.VideoCapture``. Network feeds fail routinely — an ROV surfaces,
    a switch reboots — so a dropped read triggers a bounded reconnect rather
    than ending the session. A file that reaches its end is *not* an error and
    can optionally loop, which is what makes a recorded clip usable as a stand-in
    camera for a demo.
    """

    def __init__(self, source: str | int, loop_files: bool = False,
                 reconnect: bool = True):
        self.source = parse_source(source)
        self.description = describe_source(source)
        self.loop_files = loop_files
        self.reconnect = reconnect
        self.is_file = isinstance(self.source, str) and "://" not in str(self.source)
        self._cap = None
        self._attempts = 0

    def open(self) -> None:
        cv2 = require("cv2", hint="pip install -e '.[cv]'")
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open {self.description}: {self.source!r}. "
                "For a USB camera try index 0 or 1; for RTSP check the URL and "
                "that the host is reachable.")
        LOGGER.info("Opened %s: %s", self.description, self.source)
        self._attempts = 0

    def read(self) -> np.ndarray | None:
        """Next frame as RGB, or None when the source is finished."""
        cv2 = require("cv2")
        if self._cap is None:
            self.open()
        ok, frame = self._cap.read()
        if ok:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.is_file:
            if self.loop_files:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if ok:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None                      # end of file is not a failure

        if not self.reconnect:
            return None
        self._attempts += 1
        if self._attempts > MAX_RECONNECT_ATTEMPTS:
            LOGGER.error("%s: giving up after %d reconnect attempts",
                         self.description, MAX_RECONNECT_ATTEMPTS)
            return None
        LOGGER.warning("%s: read failed, reconnecting (%d/%d)",
                       self.description, self._attempts, MAX_RECONNECT_ATTEMPTS)
        self.release()
        time.sleep(RECONNECT_DELAY_S)
        try:
            self.open()
        except RuntimeError as exc:
            LOGGER.warning("reconnect failed: %s", exc)
        return np.zeros((1, 1, 3), dtype=np.uint8) if self._cap is None else self.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "LiveSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Per-frame result
# --------------------------------------------------------------------------- #
@dataclass
class LiveFrame:
    """One processed frame and everything an operator needs to judge it."""
    index: int
    timestamp: float
    image: np.ndarray                       # annotated RGB
    raw_detections: list[BBox] = field(default_factory=list)
    confirmed: list[BBox] = field(default_factory=list)
    latency_ms: float = 0.0
    ood: bool | None = None
    ood_score: float | None = None
    method: str = ""
    # Distance travelled along the net since the session started, when odometry
    # is on. None means "not tracked" — never 0.0, which would read as "here".
    along_m: float | None = None

    @property
    def has_confirmed(self) -> bool:
        return bool(self.confirmed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "timestamp": self.timestamp, "method": self.method,
            "latency_ms": round(self.latency_ms, 1),
            "raw": len(self.raw_detections), "confirmed": len(self.confirmed),
            "ood": self.ood, "ood_score": self.ood_score,
            "detections": [
                {"bbox": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)],
                 "score": round(b.score, 3), "class": b.class_name}
                for b in self.confirmed],
        }


# --------------------------------------------------------------------------- #
# Inference over a stream
# --------------------------------------------------------------------------- #
class LiveInspector:
    """Runs detection, temporal confirmation and OOD gating over frames.

    Parameters
    ----------
    inspector : NetInspector
        The shared inference facade, so live behaviour matches batch exactly.
    method, conf : str, float
        Detector and confidence threshold.
    min_hits, max_age : int
        Temporal confirmation: a detection must persist ``min_hits`` frames to
        be reported, and survives ``max_age`` frames of absence.
    ood_model : optional
        A loaded PatchCore model; when given, frames are scored and flagged.
    ood_every : int
        Score every Nth frame rather than all of them. The gate answers "is this
        domain familiar", which changes on the scale of a scene, not a frame —
        so running it every frame mostly buys latency. The last verdict is
        carried forward between checks.
    """

    def __init__(self, inspector: Any, method: str = "yolo", conf: float = 0.25,
                 min_hits: int = 3, max_age: int = 5, ood_model: Any = None,
                 draw: bool = True, ood_every: int = 1,
                 odometry: bool = False, standoff_m: float = 0.6):
        from .temporal import TemporalConfig, Tracker

        self.inspector = inspector
        self.method = method
        self.conf = conf
        self.draw = draw
        # Live odometry places a confirmed defect on the net instead of only in
        # a frame. Scale comes from a DECLARED standoff, because a bare camera
        # feed carries no telemetry — so a position is only as good as that
        # number, and every consumer is told so rather than left to assume.
        self.odometry = bool(odometry)
        self.standoff_m = float(standoff_m)
        self.along_m: float | None = 0.0 if odometry else None
        self._prev_gray = None
        self.tracker = Tracker(TemporalConfig(min_hits=min_hits, max_age=max_age))
        self.ood_model = ood_model
        self.ood_every = max(1, int(ood_every))
        self._processed = 0
        self._last_ood: tuple[bool | None, float | None] = (None, None)
        self._gate = None
        if ood_model is not None:
            from .ood_gate import OODGate
            self._gate = OODGate(threshold=getattr(ood_model, "threshold", None))

    def process(self, frame_rgb: np.ndarray, index: int = 0) -> LiveFrame:
        """Run one frame end to end."""
        from .visualize import overlay_boxes

        t0 = time.perf_counter()
        result = self.inspector.predict(frame_rgb, method=self.method, conf=self.conf)
        confirmed = self.tracker.update(result.boxes)
        latency = (time.perf_counter() - t0) * 1000.0

        ood_flag, ood_score = self._last_ood
        if self._gate is not None and self.ood_model is not None:
            if self._processed % self.ood_every == 0:
                try:
                    from .ood_gate import OODGate
                    ood_score = float(OODGate.frame_score(frame_rgb, self.ood_model))
                    ood_flag = bool(self._gate.flag(ood_score))
                    self._last_ood = (ood_flag, ood_score)
                except Exception as exc:  # a gate fault must not stop the feed
                    LOGGER.debug("OOD scoring failed on frame %d: %s", index, exc)
        self._processed += 1

        if self.odometry:
            self._advance_odometry(frame_rgb)

        image = frame_rgb
        if self.draw:
            image = (result.heatmap if result.heatmap is not None
                     else overlay_boxes(frame_rgb, preds=confirmed))
        return LiveFrame(index=index, timestamp=time.time(), image=image,
                         raw_detections=list(result.boxes), confirmed=list(confirmed),
                         latency_ms=latency, ood=ood_flag, ood_score=ood_score,
                         method=self.method, along_m=self.along_m)

    def _advance_odometry(self, frame_rgb: np.ndarray) -> None:
        """Integrate along-track travel from feature motion between frames.

        Reuses :mod:`netinspect.mapping` rather than a second implementation, so
        a live position and an offline one are produced by the same code. An
        unmatchable pair contributes nothing instead of guessing — the position
        stalls, which is visible, rather than drifting invisibly.
        """
        from .mapping import ScaleCalibration, estimate_motion

        cv2 = optional_import("cv2")
        if cv2 is None:
            self.odometry = False
            LOGGER.warning("Live odometry needs OpenCV; disabling it for this session.")
            return
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self._prev_gray is not None:
            try:
                motion = estimate_motion(self._prev_gray, gray)
            except Exception as exc:              # never let odometry kill the feed
                LOGGER.debug("Live odometry failed: %s", exc)
                motion = None
            if motion is not None:
                mm_per_px = ScaleCalibration(
                    MM_PER_PX_AT_1M, 1.0, 0, 0, 0,
                    note="live: declared standoff").mm_per_px(self.standoff_m)
                self.along_m = (self.along_m or 0.0) + abs(motion.dx_px) * mm_per_px / 1000.0
        self._prev_gray = gray

    def run(self, source: LiveSource, max_frames: int | None = None,
            every_n: int = 1) -> Iterator[LiveFrame]:
        """Iterate processed frames from a source.

        ``every_n`` skips frames before inference — the simple way to keep a
        slow model roughly in step with a fast camera when a background thread
        is not wanted.
        """
        index = 0
        produced = 0
        while max_frames is None or produced < max_frames:
            frame = source.read()
            if frame is None:
                break
            index += 1
            if every_n > 1 and index % every_n:
                continue
            yield self.process(frame, index)
            produced += 1


# --------------------------------------------------------------------------- #
# Threaded session (what the web console drives)
# --------------------------------------------------------------------------- #
class LiveSession:
    """Background capture + inference with a latest-frame buffer.

    The point of the two threads is backlog avoidance. Capture never waits for
    inference; inference always picks up the most recent frame and discards the
    rest. On a feed the model cannot keep up with, the result is a lower
    *effective* frame rate showing current reality — rather than a smooth
    stream showing the past, which is the failure mode that makes naive live
    demos useless.
    """

    def __init__(self, source: str | int, live_inspector: LiveInspector,
                 loop_files: bool = True, on_event: Callable[[LiveFrame], None] | None = None):
        self.source_spec = source
        self.description = describe_source(source)
        self.inspector = live_inspector
        self.loop_files = loop_files
        self.on_event = on_event

        self._source: LiveSource | None = None
        self._latest: LiveFrame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._pending: np.ndarray | None = None
        self._pending_lock = threading.Lock()

        self.started_at: float | None = None
        self.frames_captured = 0
        self.frames_inferred = 0
        self.frames_dropped = 0
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.error: str | None = None
        self._confirmed_seen: set[int] = set()

    # -- lifecycle ----------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("session already running")
        self._source = LiveSource(self.source_spec, loop_files=self.loop_files)
        self._source.open()                  # fail loudly here, not in a thread
        self._stop.clear()
        self.started_at = time.time()
        self.error = None
        self._threads = [
            threading.Thread(target=self._capture_loop, name="live-capture", daemon=True),
            threading.Thread(target=self._infer_loop, name="live-infer", daemon=True),
        ]
        for t in self._threads:
            t.start()
        LOGGER.info("Live session started on %s", self.description)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads = []
        if self._source is not None:
            self._source.release()
            self._source = None
        LOGGER.info("Live session stopped after %d frames", self.frames_inferred)

    # -- threads ------------------------------------------------------------
    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._source.read() if self._source else None
                if frame is None:
                    self.error = "source ended"
                    break
                self.frames_captured += 1
                with self._pending_lock:
                    if self._pending is not None:
                        self.frames_dropped += 1     # inference is behind; drop the old one
                    self._pending = frame
        except Exception as exc:                      # pragma: no cover - thread guard
            self.error = str(exc)
            LOGGER.exception("capture loop failed")
        finally:
            self._stop.set()

    def _infer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._pending_lock:
                    frame, self._pending = self._pending, None
                if frame is None:
                    time.sleep(0.005)
                    continue
                result = self.inspector.process(frame, index=self.frames_inferred + 1)
                self.frames_inferred += 1
                with self._lock:
                    self._latest = result
                self._record_events(result)
        except Exception as exc:                      # pragma: no cover - thread guard
            self.error = str(exc)
            LOGGER.exception("inference loop failed")
        finally:
            self._stop.set()

    def _record_events(self, frame: LiveFrame) -> None:
        """Log one event per newly confirmed track, not per frame.

        Deduplicating on the tracker's stable track id is what turns a stream of
        per-frame boxes into an operator-sized number of alerts: a defect the ROV
        drifts past over 90 frames is one event, not ninety.
        """
        for track_id, box in self.inspector.tracker.confirmed_tracks():
            if track_id in self._confirmed_seen:
                continue
            self._confirmed_seen.add(track_id)
            self.events.append({
                "track_id": track_id, "frame": frame.index,
                "timestamp": frame.timestamp, "method": frame.method,
                "score": round(box.score, 3),
                "bbox": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                "ood": frame.ood,
                # Where on the net, not just which frame — None when odometry
                # is off, so a consumer can tell "not tracked" from "at 0 m".
                "along_m": (round(frame.along_m, 3)
                            if frame.along_m is not None else None),
                "hits": getattr(box, "hits", 1),
            })
            if self.on_event:
                self.on_event(frame)

    # -- readers ------------------------------------------------------------
    def latest(self) -> LiveFrame | None:
        with self._lock:
            return self._latest

    def status(self) -> dict[str, Any]:
        frame = self.latest()
        elapsed = (time.time() - self.started_at) if self.started_at else 0.0
        return {
            "running": self.running,
            "source": str(self.source_spec),
            "source_kind": self.description,
            "method": self.inspector.method,
            "conf": self.inspector.conf,
            "elapsed_s": round(elapsed, 1),
            "frames_captured": self.frames_captured,
            "frames_inferred": self.frames_inferred,
            "frames_dropped": self.frames_dropped,
            "inference_fps": round(self.frames_inferred / elapsed, 2) if elapsed > 0 else 0.0,
            "capture_fps": round(self.frames_captured / elapsed, 2) if elapsed > 0 else 0.0,
            "latency_ms": round(frame.latency_ms, 1) if frame else None,
            "confirmed_now": len(frame.confirmed) if frame else 0,
            "events": len(self.events),
            "recent_events": list(self.events)[-10:],
            "ood": frame.ood if frame else None,
            "odometry": self.inspector.odometry,
            "along_m": (round(self.inspector.along_m, 2)
                        if self.inspector.along_m is not None else None),
            "standoff_m": self.inspector.standoff_m if self.inspector.odometry else None,
            "error": self.error,
            "disclaimer": (
                "Prototype: models were trained on SYNTHETIC damage. Treat live "
                "detections as human-review triage, not verified alarms. Recall "
                "on real damage is unmeasured."),
        }


__all__ = ["LiveSource", "LiveInspector", "LiveSession", "LiveFrame",
           "parse_source", "describe_source",
           "MAX_EVENTS", "RECONNECT_DELAY_S", "MAX_RECONNECT_ATTEMPTS"]

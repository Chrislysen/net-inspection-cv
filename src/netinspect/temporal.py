"""Temporal reasoning: confirm detections that persist across video frames.

Single-frame detectors fire on transient clutter — a fish swimming past, a glint
of light, a momentary motion blur. **Real damage is static relative to the net**,
so it persists across consecutive frames while clutter does not. A lightweight
IoU tracker (a mini-SORT) associates detections frame-to-frame and only
**confirms** a track once it has been seen in ``min_hits`` frames, which removes
flicker false positives while keeping persistent damage.

This is exactly the kind of cheap, high-value post-processing an operator-facing
inspection tool needs: far fewer false alarms on real footage, at the cost of a
few frames' latency before a detection is confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .utils import BBox, iou_xyxy


@dataclass
class TemporalConfig:
    iou_match: float = 0.2     # min IoU to associate a detection with a track
    min_hits: int = 3          # frames a track must be seen before it is confirmed
    max_age: int = 2           # frames a track survives without a match
    conf: float = 0.25         # per-detection confidence gate


@dataclass
class _Track:
    track_id: int
    box: BBox
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list[BBox] = field(default_factory=list)


class Tracker:
    """Greedy IoU tracker that confirms persistent detections."""

    def __init__(self, cfg: TemporalConfig | None = None):
        self.cfg = cfg or TemporalConfig()
        self._tracks: list[_Track] = []
        self._next_id = 0

    def update(self, detections: list[BBox]) -> list[BBox]:
        """Advance one frame; return the currently-visible CONFIRMED detections."""
        cfg = self.cfg
        dets = [d for d in detections if d.score >= cfg.conf]

        # Greedy association by descending detection score.
        unmatched = set(range(len(dets)))
        for track in self._tracks:
            best_iou, best_j = cfg.iou_match, -1
            for j in unmatched:
                i = iou_xyxy(track.box.to_list(), dets[j].to_list())
                if i >= best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0:
                track.box = dets[best_j]
                track.hits += 1
                track.misses = 0
                track.history.append(dets[best_j])
                if track.hits >= cfg.min_hits:
                    track.confirmed = True
                unmatched.discard(best_j)
            else:
                track.misses += 1

        # New tracks for unmatched detections (confirmed immediately if a single
        # hit already meets min_hits).
        for j in unmatched:
            track = _Track(self._next_id, dets[j], history=[dets[j]])
            track.confirmed = track.hits >= cfg.min_hits
            self._tracks.append(track)
            self._next_id += 1

        # Drop stale tracks.
        self._tracks = [t for t in self._tracks if t.misses <= cfg.max_age]

        # Confirmed tracks that were matched this frame (misses == 0).
        return [t.box for t in self._tracks if t.confirmed and t.misses == 0]

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)


def filter_sequence(per_frame_detections: list[list[BBox]],
                    cfg: TemporalConfig | None = None) -> list[list[BBox]]:
    """Apply temporal confirmation to a whole sequence of per-frame detections."""
    tracker = Tracker(cfg)
    return [tracker.update(dets) for dets in per_frame_detections]

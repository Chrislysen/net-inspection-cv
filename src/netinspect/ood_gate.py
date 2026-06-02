"""Uncertainty / out-of-distribution gate — know when to defer to a human.

A deployed inspector must recognise when a frame is unlike anything it was trained
on (new site, turbidity, lighting, equipment in view) and **route it to human
review** instead of silently auto-deciding. This is the safety net that makes a
not-yet-certified detector usable in practice.

The PatchCore anomaly model already produces a frame-level *distance-from-normal-net*
score. Thresholding it gives a cheap, **detector-agnostic** OOD gate that sits in
front of any method: calibrate the threshold on in-distribution frames (e.g. the
95th percentile of training-clip scores), then flag any frame scoring above it.

Honesty: this flags *distribution shift*, not "is there damage" — a high score
means "unusual, look at it", which is exactly the conservative behaviour wanted
when the model is out of its depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import get_logger

LOGGER = get_logger()


@dataclass
class OODGate:
    """A calibrated frame-level out-of-distribution gate.

    ``threshold`` is an absolute anomaly-score cutoff; frames at or above it are
    flagged for human review. Use :meth:`calibrate` to set it from in-distribution
    scores at a chosen percentile (a low expected false-flag rate on normal data).
    """
    threshold: float
    percentile: float = 95.0

    @staticmethod
    def frame_score(image_rgb: np.ndarray, patchcore_model) -> float:
        """Frame-level OOD score = max patch distance-from-normal (PatchCore)."""
        from .patchcore import score_image
        return float(score_image(image_rgb, patchcore_model).max_score)

    @classmethod
    def calibrate(cls, in_dist_scores: list[float], percentile: float = 95.0) -> "OODGate":
        if not in_dist_scores:
            raise ValueError("Need in-distribution scores to calibrate the gate.")
        thr = float(np.percentile(in_dist_scores, percentile))
        LOGGER.info("OOD gate calibrated: threshold=%.3f at p%.0f of %d in-dist frames",
                    thr, percentile, len(in_dist_scores))
        return cls(threshold=thr, percentile=percentile)

    def flag(self, score: float) -> bool:
        """True -> frame is out-of-distribution; route to human review."""
        return score >= self.threshold

    def flag_rate(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        return round(sum(1 for s in scores if self.flag(s)) / len(scores), 3)

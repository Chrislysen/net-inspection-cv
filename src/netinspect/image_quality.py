"""No-reference capture-quality metrics for inspection frames.

Measured on this repo's SOLAQUA frames, **capture quality — not standoff
distance and not recording day — is the strongest single correlate of
false-alarm behaviour**, and its sign depends on the model:

===========  =========================  ==================================
Model        corr(sharpness, false alarm)  Failure mode
===========  =========================  ==================================
det_v1       +0.60 (p ~ 1e-62)          fires on sharp, high-contrast detail
seg_gpu      -0.11 (p ~ 4e-3)           fires on blurred, low-contrast frames
===========  =========================  ==================================

The two failure modes point in opposite directions, which is the mechanism
behind the det-and-seg agreement ensemble already in this repo: requiring both
models to agree cancels two errors that occur in different capture regimes.

The metrics here are deliberately cheap, classical and interpretable rather
than learned. They need no reference image, run in milliseconds, and can
therefore gate a live inspection feed. ``variance_of_laplacian`` is the
standard Pech-Pacheco focus measure; it is *relative*, so values are only
comparable across frames from the same camera and resolution — which is the
case within a SOLAQUA clip but not necessarily between different vehicles.

Caveat
------
These are correlations on undamaged real net, where every detection is a known
false positive. They characterise false-alarm behaviour only. Whether recall on
real damage degrades in the same regimes cannot be answered without real
labelled damage, and is not claimed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .utils import get_logger, optional_import

LOGGER = get_logger()


@dataclass
class QualityMetrics:
    """No-reference quality descriptors for a single frame.

    Attributes
    ----------
    sharpness : float
        Variance of the Laplacian. Higher = more high-frequency detail. Falls
        with motion blur, defocus and light attenuation.
    contrast : float
        Standard deviation of luminance.
    brightness : float
        Mean luminance in [0, 255].
    saturation : float
        Mean chroma; a rough turbidity/colour-cast indicator underwater.
    dark_fraction : float
        Share of pixels below :data:`DARK_LEVEL`. Relevant here because the
        classical baseline keys on absolute darkness as a see-through cue.
    """
    sharpness: float
    contrast: float
    brightness: float
    saturation: float
    dark_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


DARK_LEVEL = 40
METRIC_NAMES = ("sharpness", "contrast", "brightness", "saturation", "dark_fraction")


def _luminance(image_rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma as float32, without requiring OpenCV."""
    arr = np.asarray(image_rgb)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def variance_of_laplacian(image_rgb: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian of the luminance channel.

    Uses OpenCV when available and falls back to an equivalent 4-neighbour
    convolution otherwise, so the metric is available in the minimal
    environment the rest of the package targets.
    """
    gray = _luminance(image_rgb)
    cv2 = optional_import("cv2")
    if cv2 is not None:
        # Source is float32, so the destination depth must be CV_32F — OpenCV
        # has no float32 -> float64 Laplacian kernel.
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    lap = (-4.0 * gray
           + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
    return float(lap[1:-1, 1:-1].var())


def compute(image_rgb: np.ndarray) -> QualityMetrics:
    """Compute all quality metrics for one RGB uint8 frame."""
    arr = np.asarray(image_rgb)
    gray = _luminance(arr)
    if arr.ndim == 3:
        saturation = float((arr.max(axis=2).astype(np.float32)
                            - arr.min(axis=2).astype(np.float32)).mean())
    else:
        saturation = 0.0
    return QualityMetrics(
        sharpness=variance_of_laplacian(arr),
        contrast=float(gray.std()),
        brightness=float(gray.mean()),
        saturation=saturation,
        dark_fraction=float((gray < DARK_LEVEL).mean()),
    )


# --------------------------------------------------------------------------- #
# Quality gating
# --------------------------------------------------------------------------- #
@dataclass
class QualityBand:
    """A validated capture-quality range for one model.

    Both bounds matter and they are not symmetric in meaning:

    ``sharpness_min``
        Below this the frame is degraded (motion blur, light attenuation,
        turbidity). Segmentation models in this repo false-alarm here.
    ``sharpness_max``
        Above this the frame resolves fine structure — biofouling strands,
        mesh knots — that the detector confuses with damage.

    An unbounded side is expressed as ``None`` so the band survives a JSON
    round-trip without infinities.
    """
    sharpness_min: float | None = None
    sharpness_max: float | None = None
    contrast_min: float | None = None
    contrast_max: float | None = None
    model: str | None = None
    evidence: dict[str, Any] | None = None

    def check(self, m: QualityMetrics) -> tuple[bool, list[str]]:
        """Return ``(within_band, reasons_it_is_not)``."""
        reasons: list[str] = []
        if self.sharpness_min is not None and m.sharpness < self.sharpness_min:
            reasons.append(f"sharpness {m.sharpness:.0f} below validated minimum "
                           f"{self.sharpness_min:.0f} (blur / light loss)")
        if self.sharpness_max is not None and m.sharpness > self.sharpness_max:
            reasons.append(f"sharpness {m.sharpness:.0f} above validated maximum "
                           f"{self.sharpness_max:.0f} (fine structure resolved)")
        if self.contrast_min is not None and m.contrast < self.contrast_min:
            reasons.append(f"contrast {m.contrast:.1f} below validated minimum "
                           f"{self.contrast_min:.1f}")
        if self.contrast_max is not None and m.contrast > self.contrast_max:
            reasons.append(f"contrast {m.contrast:.1f} above validated maximum "
                           f"{self.contrast_max:.1f}")
        return (not reasons), reasons

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QualityBand":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def fit_band(values: np.ndarray, events: np.ndarray, target_rate: float = 0.05,
             n_bins: int = 8, model: str | None = None,
             metric: str = "sharpness") -> QualityBand:
    """Fit the widest contiguous quantile range meeting a false-alarm target.

    Bins on quantiles rather than fixed widths because these metrics are
    heavily skewed. A bin qualifies only when the *upper* bound of its Wilson
    interval clears ``target_rate``, so a handful of clean frames cannot
    certify a range.

    Returns a band with ``evidence.fitted = False`` (and no bounds) when no
    contiguous range qualifies, rather than a permissive default.
    """
    from .envelope import wilson_ci

    values = np.asarray(values, dtype=float)
    events = np.asarray(events, dtype=int)
    ok = np.isfinite(values)
    values, events = values[ok], events[ok]
    if values.size == 0:
        return QualityBand(model=model, evidence={"fitted": False, "note": "no data"})

    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return QualityBand(model=model, evidence={"fitted": False,
                                                  "note": "metric is constant"})
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (values >= lo) & (values <= hi if hi == edges[-1] else values < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        k = int(events[sel].sum())
        _, upper = wilson_ci(k, n)
        rows.append({"lo": float(lo), "hi": float(hi), "n": n, "k": k,
                     "rate": k / n, "ci_upper": upper, "ok": upper <= target_rate})

    best: list[dict] = []
    run: list[dict] = []
    for r in rows:
        if r["ok"]:
            run.append(r)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []

    if not best:
        return QualityBand(model=model, evidence={
            "fitted": False, "metric": metric, "target_rate": target_rate,
            "note": "No contiguous quality range met the target at 95% confidence.",
            "bins": rows})

    n = sum(r["n"] for r in best)
    k = sum(r["k"] for r in best)
    lo_bound = None if best[0] is rows[0] else round(best[0]["lo"], 2)
    hi_bound = None if best[-1] is rows[-1] else round(best[-1]["hi"], 2)
    kwargs = {f"{metric}_min": lo_bound, f"{metric}_max": hi_bound}
    return QualityBand(model=model, evidence={
        "fitted": True, "metric": metric, "target_rate": target_rate,
        "frames": n, "false_alarm_frames": k,
        "measured_rate": round(k / n, 4),
        "bins": best,
        "caveat": ("Fitted on undamaged real net: bounds false alarms only. Recall on "
                   "real damage in these regimes is unmeasured."),
    }, **kwargs)  # type: ignore[arg-type]


def batch_metrics(images) -> list[QualityMetrics]:
    """Compute metrics for an iterable of RGB frames."""
    return [compute(im) for im in images]


__all__ = ["DARK_LEVEL", "METRIC_NAMES", "QualityMetrics", "QualityBand",
           "variance_of_laplacian", "compute", "fit_band", "batch_metrics"]

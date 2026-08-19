"""Inspection operating envelope: when is a detection worth believing?

A detector's measured false-alarm rate is only valid under the conditions it
was measured in. For net inspection those conditions are set by the vehicle:
how far it flew from the net, how fast it swept, and whether its net-plane
estimate was locked. This module turns that idea into three concrete pieces:

:class:`EnvelopeSpec`
    The declared valid operating range, with the evidence behind it.
:class:`EnvelopeGate`
    Per-frame verdict — ``in_envelope`` / ``out_of_envelope`` / ``unknown`` —
    with a human-readable reason, so an out-of-envelope frame can be routed to
    "re-fly this section" instead of silently producing an untrusted result.
Statistics
    Binomial helpers plus the dose-response, matched-band and envelope-fitting
    routines used by ``scripts/analyze_operating_envelope.py``.

Design note
-----------
The gate deliberately reports ``unknown`` rather than ``in_envelope`` when
telemetry is missing. A frame with no net-plane lock has not been shown to be
inside the envelope; treating absence of evidence as compliance is exactly the
failure mode this module exists to prevent.

This complements :mod:`netinspect.ood_gate`, which flags frames that *look*
unfamiliar to the model. The envelope gate flags frames that were *captured*
outside validated conditions. They catch different failures: a frame can look
perfectly ordinary and still have been shot from twice the validated range.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .utils import get_logger

LOGGER = get_logger()

IN_ENVELOPE = "in_envelope"
OUT_OF_ENVELOPE = "out_of_envelope"
UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Binomial statistics
# --------------------------------------------------------------------------- #
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) approximation because false-alarm rates
    here are small and per-bin counts are sometimes tiny — regimes where Wald
    intervals are badly calibrated and can extend outside [0, 1].

    Parameters
    ----------
    successes, n : int
        Events and trials.
    z : float
        Normal quantile; 1.96 gives a 95% interval.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)``, or ``(nan, nan)`` when ``n == 0``.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion_stat(events: Sequence[bool] | Sequence[int]) -> dict[str, Any]:
    """Summarise a boolean sequence as ``{n, k, rate, ci95}``."""
    seq = list(events)
    n = len(seq)
    k = int(sum(1 for x in seq if x))
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci95": None}
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "rate": round(k / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def intervals_overlap(a: Sequence[float] | None, b: Sequence[float] | None) -> bool:
    """True when two confidence intervals overlap (or either is missing)."""
    if not a or not b:
        return True
    return a[1] >= b[0] and b[1] >= a[0]


# --------------------------------------------------------------------------- #
# Envelope specification and gate
# --------------------------------------------------------------------------- #
@dataclass
class EnvelopeSpec:
    """A declared valid operating range for an inspection model.

    Attributes
    ----------
    standoff_min_m, standoff_max_m : float
        Validated range of distance from the net.
    speed_max_ms : float, optional
        Maximum net-relative sweep speed. ``None`` disables the check.
    require_lock : bool
        Require the net-plane estimator to report a lock. When the estimate is
        unlocked, standoff is not a measurement and the frame is ``unknown``.
    model : str, optional
        Which model this envelope was fitted for — an envelope is a property of
        a *model on data*, not of the vehicle alone.
    evidence : dict
        Provenance: measured rate inside the envelope, frame counts, the target
        it was fitted to, and the caveats that apply.
    """
    standoff_min_m: float = 0.0
    standoff_max_m: float = float("inf")
    speed_max_ms: float | None = None
    require_lock: bool = True
    model: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if math.isinf(d["standoff_max_m"]):
            d["standoff_max_m"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvelopeSpec":
        d = dict(d)
        if d.get("standoff_max_m") is None:
            d["standoff_max_m"] = float("inf")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def describe(self) -> str:
        hi = "inf" if math.isinf(self.standoff_max_m) else f"{self.standoff_max_m:.2f}"
        parts = [f"standoff {self.standoff_min_m:.2f}-{hi} m"]
        if self.speed_max_ms is not None:
            parts.append(f"sweep speed <= {self.speed_max_ms:.2f} m/s")
        if self.require_lock:
            parts.append("net-plane lock required")
        return "; ".join(parts)


@dataclass
class EnvelopeVerdict:
    """Per-frame envelope decision."""
    status: str
    reasons: list[str] = field(default_factory=list)
    standoff_m: float | None = None
    speed_ms: float | None = None

    @property
    def trusted(self) -> bool:
        """True only for frames positively shown to be inside the envelope."""
        return self.status == IN_ENVELOPE

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons),
                "standoff_m": self.standoff_m, "speed_ms": self.speed_ms,
                "trusted": self.trusted}


class EnvelopeGate:
    """Classify frames against an :class:`EnvelopeSpec`.

    Examples
    --------
    >>> spec = EnvelopeSpec(standoff_min_m=0.4, standoff_max_m=0.9, speed_max_ms=0.2)
    >>> EnvelopeGate(spec).check(standoff_m=1.4, speed_ms=0.28, locked=True).status
    'out_of_envelope'
    """

    def __init__(self, spec: EnvelopeSpec | None = None):
        self.spec = spec or EnvelopeSpec()

    def check(self, standoff_m: float | None = None, speed_ms: float | None = None,
              locked: bool | None = None) -> EnvelopeVerdict:
        """Return a verdict for one frame's capture conditions."""
        s = self.spec
        reasons: list[str] = []

        # A spec whose bounds are NaN comes from a fit that failed to find any
        # qualifying range. Comparisons against NaN are always False, so without
        # this guard such a spec would silently trust every frame — the exact
        # permissive default this class exists to prevent.
        if math.isnan(s.standoff_min_m) or math.isnan(s.standoff_max_m):
            return EnvelopeVerdict(
                UNKNOWN, ["no validated operating envelope for this model"],
                standoff_m, speed_ms)

        # `not locked`, not `locked is False`. The parameter defaults to None,
        # meaning "no lock telemetry for this frame" — which is precisely the
        # case the docstring above calls unknown, yet `is False` let it through
        # as trusted. Any falsy non-bool (0, 0.0) was waved through too.
        if s.require_lock and not locked:
            return EnvelopeVerdict(UNKNOWN, ["net-plane estimate not locked"],
                                   standoff_m, speed_ms)
        if standoff_m is None or (isinstance(standoff_m, float) and math.isnan(standoff_m)):
            return EnvelopeVerdict(UNKNOWN, ["no standoff telemetry for this frame"],
                                   standoff_m, speed_ms)

        if standoff_m < s.standoff_min_m:
            reasons.append(f"standoff {standoff_m:.2f} m below validated minimum "
                           f"{s.standoff_min_m:.2f} m")
        if standoff_m > s.standoff_max_m:
            reasons.append(f"standoff {standoff_m:.2f} m above validated maximum "
                           f"{s.standoff_max_m:.2f} m")
        if s.speed_max_ms is not None and speed_ms is not None \
                and not math.isnan(speed_ms) and speed_ms > s.speed_max_ms:
            reasons.append(f"sweep speed {speed_ms:.2f} m/s above validated maximum "
                           f"{s.speed_max_ms:.2f} m/s")

        status = OUT_OF_ENVELOPE if reasons else IN_ENVELOPE
        return EnvelopeVerdict(status, reasons, standoff_m, speed_ms)

    def summarise(self, verdicts: Iterable[EnvelopeVerdict]) -> dict[str, Any]:
        """Aggregate verdicts into an inspection-validity summary.

        ``compliance`` is the share of frames positively inside the envelope.
        ``unknown`` frames count against it, because an inspection you cannot
        verify is not an inspection you can sign off.
        """
        vs = list(verdicts)
        n = len(vs)
        counts = {IN_ENVELOPE: 0, OUT_OF_ENVELOPE: 0, UNKNOWN: 0}
        for v in vs:
            counts[v.status] = counts.get(v.status, 0) + 1
        reasons: dict[str, int] = {}
        for v in vs:
            for r in v.reasons:
                key = r.split(" ")[0] + " " + r.split(" ")[1] if " " in r else r
                reasons[key] = reasons.get(key, 0) + 1
        return {
            "frames": n,
            "counts": counts,
            "compliance": round(counts[IN_ENVELOPE] / n, 4) if n else None,
            "spec": self.spec.to_dict(),
            "spec_human": self.spec.describe(),
            "top_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:5]),
            "verdict": _inspection_verdict(counts[IN_ENVELOPE] / n if n else 0.0),
        }


def _inspection_verdict(compliance: float) -> str:
    if compliance >= 0.95:
        return "accept — inspection flown inside validated conditions"
    if compliance >= 0.80:
        return "accept with note — a minority of frames fell outside validated conditions"
    return "re-fly recommended — most frames fell outside validated conditions"


# --------------------------------------------------------------------------- #
# Fitting an envelope from measured data
# --------------------------------------------------------------------------- #
def dose_response(standoff: Sequence[float], events: Sequence[int],
                  edges: Sequence[float]) -> list[dict[str, Any]]:
    """False-alarm rate per standoff bin.

    Parameters
    ----------
    standoff : sequence of float
        Per-frame standoff in metres.
    events : sequence of int
        1 when the frame produced at least one (false) detection.
    edges : sequence of float
        Bin edges, ascending.

    Returns
    -------
    list of dict
        One entry per non-empty bin with ``n``, ``k``, ``rate`` and ``ci95``.
    """
    rows: list[dict[str, Any]] = []
    pairs = [(float(s), int(e)) for s, e in zip(standoff, events)
             if s is not None and not (isinstance(s, float) and math.isnan(s))]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [e for s, e in pairs if lo <= s < hi]
        if not sel:
            continue
        rows.append({"standoff_lo": round(float(lo), 3),
                     "standoff_hi": round(float(hi), 3),
                     **proportion_stat(sel)})
    return rows


def fit_envelope(rows: Sequence[dict[str, Any]], target_rate: float,
                 model: str | None = None) -> EnvelopeSpec:
    """Fit the widest contiguous standoff range meeting a false-alarm target.

    A bin qualifies only when the *upper* bound of its 95% interval is at or
    below ``target_rate`` — requiring the point estimate alone would let a bin
    of three frames with zero events look like proof.

    Returns an :class:`EnvelopeSpec` whose ``evidence`` records what was fitted
    and on how much data; when no bin qualifies, the returned spec has an empty
    range and an explanatory note rather than a silently permissive default.
    """
    qualifying = [r for r in rows if r.get("ci95") and r["ci95"][1] <= target_rate]
    if not qualifying:
        return EnvelopeSpec(
            standoff_min_m=float("nan"), standoff_max_m=float("nan"), model=model,
            evidence={"target_rate": target_rate, "fitted": False,
                      "note": "No standoff bin met the target at 95% confidence."})

    best: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    qual_ids = {id(r) for r in qualifying}
    for r in rows:
        if id(r) in qual_ids:
            run.append(r)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []

    n = sum(r["n"] for r in best)
    k = sum(r["k"] for r in best)
    lo, hi = wilson_ci(k, n)
    return EnvelopeSpec(
        standoff_min_m=best[0]["standoff_lo"],
        standoff_max_m=best[-1]["standoff_hi"],
        model=model,
        evidence={
            "target_rate": target_rate,
            "fitted": True,
            "frames": n,
            "false_alarm_frames": k,
            "measured_rate": round(k / n, 4) if n else None,
            "measured_rate_ci95": [round(lo, 4), round(hi, 4)],
            "bins": best,
            "caveat": ("Bounds false alarms on real undamaged net only. Recall on real "
                       "damage is not measured by this data and remains unvalidated."),
        })


def matched_band_comparison(groups: dict[str, Sequence[int]]) -> dict[str, Any]:
    """Compare false-alarm rates between groups already matched on conditions.

    ``groups`` maps a label (e.g. a recording day) to that group's per-frame
    event flags, restricted to a band where every group has data. Overlapping
    intervals mean the grouping variable does not explain a difference once the
    matched condition is controlled.
    """
    stats = {label: proportion_stat(events) for label, events in groups.items()}
    labels = [k for k, v in stats.items() if v["n"] > 0]
    out: dict[str, Any] = {"groups": stats}
    if len(labels) == 2:
        a, b = stats[labels[0]], stats[labels[1]]
        overlap = intervals_overlap(a["ci95"], b["ci95"])
        out["intervals_overlap"] = overlap
        out["interpretation"] = (
            "Rates overlap within 95% CI — the grouping variable is not needed to "
            "explain the difference once the matched condition is controlled."
            if overlap else
            "Rates separate — a residual effect remains beyond the matched condition.")
        try:
            from scipy.stats import fisher_exact
            _, p = fisher_exact([[a["k"], a["n"] - a["k"]], [b["k"], b["n"] - b["k"]]])
            out["fisher_p"] = round(float(p), 5)
        except Exception:
            pass
    return out


__all__ = [
    "IN_ENVELOPE", "OUT_OF_ENVELOPE", "UNKNOWN",
    "wilson_ci", "proportion_stat", "intervals_overlap",
    "EnvelopeSpec", "EnvelopeVerdict", "EnvelopeGate",
    "dose_response", "fit_envelope", "matched_band_comparison",
]

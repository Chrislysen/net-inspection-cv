"""Ocean conditions, cod thermal physics, and when a net inspection is flyable.

Two questions a farm actually asks, answered from live public data:

1. **When can we inspect?** MET Norway's ocean forecast gives hourly wave height
   and current speed at any Norwegian coordinate. Both bound ROV work — surface
   state governs launch and recovery, and ambient current competes directly with
   the vehicle's commanded sweep speed. The net-following telemetry in this repo
   shows sweeps commanded at 0.10–0.30 m/s, so a 0.25 m/s current is not a minor
   disturbance; it is the same magnitude as the manoeuvre.
2. **What is the water doing to the fish?** Sea temperature drives cod growth,
   and cod's thermal optimum *falls as the fish grows* — which is why a single
   fitted growth constant over a full production cycle misleads.

Both come from ``api.met.no`` (Norwegian Licence for Open Government Data +
CC BY 4.0; attribute "Data from MET Norway"). No key required, but the API
mandates an identifying ``User-Agent``.

The honesty boundary here
-------------------------
The thermal-growth relationships below are **published science**, cited inline.
The inspection-window thresholds are **not validated operating limits** — they
are derived from the range of conditions observed in this repo's own SOLAQUA
telemetry plus ordinary small-ROV practice. They say "these conditions resemble
the ones the models were characterised in", never "detection accuracy is
guaranteed". A real operator's limits come from their vehicle and their crew.

Examples
--------
>>> from netinspect import ocean
>>> fc = ocean.fetch_forecast(62.3065, 6.0920)          # Vartdal
>>> ocean.inspection_windows(fc)[:2]
>>> ocean.thermal_optimum_c(2000)                        # ~9 degC for a 2 kg cod
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .utils import get_logger

LOGGER = get_logger()

MET_OCEAN_URL = "https://api.met.no/weatherapi/oceanforecast/2.0/complete"
USER_AGENT = ("net-inspection-cv/0.1 "
              "(https://github.com/Chrislysen/net-inspection-cv)")

# Conditions in which this repo's models were actually characterised, taken from
# the SOLAQUA net-following telemetry (commanded sweep 0.10-0.30 m/s). These are
# reference points for "is this like the evaluated data", NOT certified limits.
SWEEP_SPEED_RANGE_MS = (0.10, 0.30)

# Small-ROV surface-condition guidance. Engineering judgement, not a standard.
WAVE_GOOD_M = 0.5
WAVE_MARGINAL_M = 1.0

# Cod thermal-optimum coefficients: T_opt = A - B * ln(W_grams).
# Bjornsson & Steinarsson, on Atlantic cod (Gadus morhua) growth vs temperature.
THERMAL_OPT_A = 15.57
THERMAL_OPT_B = 0.8426


# --------------------------------------------------------------------------- #
# MET Norway ocean forecast
# --------------------------------------------------------------------------- #
@dataclass
class OceanSample:
    """One hourly forecast step."""
    time: str
    sea_water_temperature_c: float | None = None
    sea_water_speed_ms: float | None = None
    sea_surface_wave_height_m: float | None = None
    sea_water_to_direction_deg: float | None = None
    sea_surface_wave_from_direction_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_forecast(lat: float, lon: float, timeout: int = 60,
                   user_agent: str = USER_AGENT) -> list[OceanSample]:
    """Fetch the hourly ocean forecast for one coordinate.

    Raises
    ------
    RuntimeError
        If MET rejects the request. The most common cause is a default or
        missing ``User-Agent`` — the API requires an identifying one and
        answers 403 otherwise.
    """
    params = urllib.parse.urlencode({"lat": round(lat, 4), "lon": round(lon, 4)})
    req = urllib.request.Request(f"{MET_OCEAN_URL}?{params}",
                                 headers={"User-Agent": user_agent,
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        if exc.code == 403:
            raise RuntimeError(
                "MET Norway returned 403 — it requires an identifying "
                "User-Agent naming your application and contact.") from exc
        raise

    out: list[OceanSample] = []
    for step in payload.get("properties", {}).get("timeseries", []):
        d = step.get("data", {}).get("instant", {}).get("details", {})
        out.append(OceanSample(
            time=step.get("time", ""),
            sea_water_temperature_c=d.get("sea_water_temperature"),
            sea_water_speed_ms=d.get("sea_water_speed"),
            sea_surface_wave_height_m=d.get("sea_surface_wave_height"),
            sea_water_to_direction_deg=d.get("sea_water_to_direction"),
            sea_surface_wave_from_direction_deg=d.get("sea_surface_wave_from_direction"),
        ))
    LOGGER.info("MET ocean forecast: %d hourly steps at %.4f, %.4f",
                len(out), lat, lon)
    return out


# --------------------------------------------------------------------------- #
# Inspection windows
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    """A judgement about whether one forecast hour suits an ROV net inspection."""
    time: str
    rating: str                      # good | marginal | poor | unknown
    reasons: list[str]
    wave_height_m: float | None
    current_speed_ms: float | None
    current_vs_sweep: float | None   # current as a multiple of the slowest sweep

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rate_conditions(sample: OceanSample) -> Window:
    """Rate one forecast hour for ROV net-following work.

    The current check is the informative one and is expressed relative to the
    *commanded sweep speed* rather than as an absolute number: a current at or
    above the sweep speed means the vehicle spends its control authority holding
    station instead of tracking the net, and the resulting imagery is captured
    under conditions unlike the evaluated data.
    """
    reasons: list[str] = []
    wave = sample.sea_surface_wave_height_m
    cur = sample.sea_water_speed_ms

    if wave is None and cur is None:
        return Window(sample.time, "unknown", ["no forecast values for this hour"],
                      wave, cur, None)

    rating = "good"
    if wave is not None:
        if wave > WAVE_MARGINAL_M:
            rating = "poor"
            reasons.append(f"wave height {wave:.1f} m above {WAVE_MARGINAL_M} m "
                           "— launch and recovery difficult")
        elif wave > WAVE_GOOD_M:
            rating = "marginal"
            reasons.append(f"wave height {wave:.1f} m — workable but degraded")

    ratio = None
    if cur is not None:
        ratio = cur / SWEEP_SPEED_RANGE_MS[0]
        if cur >= SWEEP_SPEED_RANGE_MS[1]:
            rating = "poor"
            reasons.append(
                f"current {cur:.2f} m/s at or above the fastest commanded sweep "
                f"({SWEEP_SPEED_RANGE_MS[1]} m/s) — net-following degraded")
        elif cur >= SWEEP_SPEED_RANGE_MS[0]:
            if rating == "good":
                rating = "marginal"
            reasons.append(
                f"current {cur:.2f} m/s comparable to the slowest commanded sweep "
                f"({SWEEP_SPEED_RANGE_MS[0]} m/s) — expect station-keeping effort")

    if not reasons:
        reasons.append("wave and current both inside the range this system was "
                       "characterised in")
    return Window(sample.time, rating, reasons, wave, cur,
                  round(ratio, 2) if ratio is not None else None)


def inspection_windows(forecast: Iterable[OceanSample]) -> list[Window]:
    """Rate every hour of a forecast."""
    return [rate_conditions(s) for s in forecast]


def best_window(windows: Iterable[Window]) -> Window | None:
    """The earliest 'good' hour, or the earliest 'marginal' one if none is good."""
    ws = list(windows)
    for target in ("good", "marginal"):
        for w in ws:
            if w.rating == target:
                return w
    return None


def summarise_windows(windows: Iterable[Window]) -> dict[str, Any]:
    """Aggregate hourly ratings into a planning summary."""
    ws = list(windows)
    if not ws:
        return {"hours": 0}
    counts: dict[str, int] = {}
    for w in ws:
        counts[w.rating] = counts.get(w.rating, 0) + 1
    best = best_window(ws)
    return {
        "hours": len(ws),
        "from": ws[0].time, "to": ws[-1].time,
        "ratings": counts,
        "good_fraction": round(counts.get("good", 0) / len(ws), 3),
        "first_good_window": best.time if best else None,
        "first_good_rating": best.rating if best else None,
        "caveat": ("Thresholds are reference points from this repo's own "
                   "telemetry and ordinary small-ROV practice, not certified "
                   "operating limits. They indicate similarity to the evaluated "
                   "conditions, not guaranteed detection performance."),
    }


# --------------------------------------------------------------------------- #
# Cod thermal physics
# --------------------------------------------------------------------------- #
def thermal_optimum_c(weight_g: float) -> float:
    """Temperature of maximum growth for an Atlantic cod of a given weight.

    ``T_opt = 15.57 - 0.8426 * ln(W)`` with W in grams (Bjornsson & Steinarsson).

    The consequence is the part that matters operationally: the optimum **falls
    as the fish grows**. A 2 g juvenile grows fastest near 15 °C, a 2 kg
    grow-out fish near 9 °C. Norwegian summer surface temperature sits above the
    optimum for market-size cod, so warm water that helps a hatchery hurts a
    grow-out pen.
    """
    if weight_g <= 0:
        raise ValueError("weight_g must be positive")
    return THERMAL_OPT_A - THERMAL_OPT_B * math.log(weight_g)


def thermal_stress(weight_g: float, temperature_c: float) -> dict[str, Any]:
    """How far a given temperature sits from optimal for this size of fish."""
    opt = thermal_optimum_c(weight_g)
    delta = temperature_c - opt
    if delta > 3:
        state = "above optimum — growth suppressed, oxygen demand elevated"
    elif delta > 1:
        state = "slightly above optimum"
    elif delta < -3:
        state = "well below optimum — growth slowed"
    elif delta < -1:
        state = "slightly below optimum"
    else:
        state = "near optimum"
    return {
        "weight_g": weight_g,
        "temperature_c": round(temperature_c, 2),
        "thermal_optimum_c": round(opt, 2),
        "delta_c": round(delta, 2),
        "state": state,
    }


def tgc_growth(initial_weight_g: float, temperatures_c: Iterable[float],
               tgc: float = 2.0, days_per_step: float = 1.0) -> list[float]:
    """Project weight with the thermal growth coefficient model.

    ``W_t^(1/3) = W_0^(1/3) + TGC/1000 * sum(T * dt)``

    Only positive temperature contributes; the standard TGC form has no term for
    growth arrest above the optimum, which is exactly the limitation
    :func:`tgc_growth_thermal_corrected` addresses.
    """
    if initial_weight_g <= 0:
        raise ValueError("initial_weight_g must be positive")
    w_cbrt = initial_weight_g ** (1 / 3)
    out: list[float] = []
    for t in temperatures_c:
        w_cbrt += (tgc / 1000.0) * max(0.0, float(t)) * days_per_step
        out.append(w_cbrt ** 3)
    return out


def tgc_growth_thermal_corrected(initial_weight_g: float,
                                 temperatures_c: Iterable[float],
                                 tgc: float = 2.0,
                                 days_per_step: float = 1.0) -> list[float]:
    """TGC projection that penalises temperature away from the size-dependent optimum.

    Plain TGC assumes growth scales linearly with temperature, so it happily
    predicts that a 4 kg cod grows faster at 16 °C than at 10 °C. It does not:
    the optimum for that fish is near 8 °C. This variant scales the daily
    increment by a triangular efficiency term centred on
    :func:`thermal_optimum_c`, recomputed as the fish grows.

    It is a **correction of shape, not a calibrated model** — the efficiency
    falloff width is a modelling choice, and no cod-farm production data was
    available to fit it. Its purpose is to show the direction and rough size of
    the bias that an uncorrected TGC carries over a full cycle. Fitting it needs
    an operator's per-cohort records.
    """
    if initial_weight_g <= 0:
        raise ValueError("initial_weight_g must be positive")
    falloff_c = 6.0   # temperature offset at which growth efficiency reaches zero
    weight = float(initial_weight_g)
    out: list[float] = []
    for t in temperatures_c:
        opt = thermal_optimum_c(weight)
        efficiency = max(0.0, 1.0 - abs(float(t) - opt) / falloff_c)
        w_cbrt = weight ** (1 / 3) + (tgc / 1000.0) * max(0.0, float(t)) \
            * efficiency * days_per_step
        weight = w_cbrt ** 3
        out.append(weight)
    return out


def degree_days(temperatures_c: Iterable[float], days_per_step: float = 1.0) -> float:
    """Accumulated thermal exposure — the standard cross-site growth clock."""
    return sum(max(0.0, float(t)) * days_per_step for t in temperatures_c)


__all__ = [
    "MET_OCEAN_URL", "USER_AGENT", "SWEEP_SPEED_RANGE_MS",
    "WAVE_GOOD_M", "WAVE_MARGINAL_M", "OceanSample", "Window",
    "fetch_forecast", "rate_conditions", "inspection_windows", "best_window",
    "summarise_windows", "thermal_optimum_c", "thermal_stress", "tgc_growth",
    "tgc_growth_thermal_corrected", "degree_days",
]

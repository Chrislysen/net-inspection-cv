"""Release gate: decide whether a model is allowed to be used, and at what threshold.

A model that has been trained is not a model that may be deployed. This module
is the difference between the two. It measures a candidate against an
**operating point the operator wrote down in advance**, and returns a verdict
that a CI job or a deployment script can act on.

Why frame-level false alarms are the primary axis
-------------------------------------------------
Detection papers optimise mAP. Net inspection does not run on mAP. A crew
watching a screen abandons a system that cries wolf, so the number that decides
whether this gets used at all is: *of the frames showing clean net, what
fraction produced an alert?* Recall is the second axis and is meaningless
without the first — a detector that fires on everything has perfect recall.

So the gate takes both, plus the rule that neither can be measured from data
that does not contain the relevant case. A test set with no clean frames cannot
produce a false-alarm rate, and this refuses to invent one. That refusal is the
feature: silence there is how a model ships on a number nobody computed.

Calibration
-----------
:func:`choose_threshold` picks the confidence threshold **on the operator's own
validation split** — the largest honest accuracy gain available without new
data, and one that transfers far worse between sites than people assume. The
default 0.25 in this repo was tuned on SOLAQUA footage; on a different farm,
camera and water it is close to arbitrary.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .utils import BBox, get_logger

LOGGER = get_logger()


@dataclass
class OperatingPoint:
    """The contract a model must satisfy before it may be used.

    Written down before evaluation, not chosen afterwards to fit the result.
    """
    conf: float = 0.25
    iou: float = 0.5
    # Fraction of CLEAN frames that produced at least one detection. This is the
    # trust number: exceed it and operators stop looking at the alerts.
    max_false_alarm_rate: float = 0.05
    # Image-level recall on frames that do contain damage.
    min_recall: float = 0.80
    min_precision: float | None = None
    # A gate computed over a handful of frames is theatre; require enough of both
    # classes for the numbers to mean anything.
    min_clean_frames: int = 50
    min_damaged_frames: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OperatingPoint":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Check:
    name: str
    value: float | None
    bound: float
    rule: str                       # "<=" or ">="
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    passed: bool
    checks: list[Check] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)
    operating_point: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["summary"] = self.summary()
        return d

    def summary(self) -> str:
        head = "PASS — may be deployed" if self.passed else "FAIL — must not be deployed"
        lines = [head]
        for c in self.checks:
            mark = "ok  " if c.passed else "FAIL"
            got = "not measurable" if c.value is None else f"{c.value:.4g}"
            lines.append(f"  [{mark}] {c.name}: {got} (needs {c.rule} {c.bound:g}) — {c.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure(preds_by_image: dict[str, list[BBox]],
            gts_by_image: dict[str, list[BBox]],
            conf: float = 0.25) -> dict[str, Any]:
    """Frame-level behaviour at one confidence threshold.

    Frames are split by their ground truth: those with no boxes are clean net and
    only a false alarm can come from them; those with boxes are the recall test.
    Both are reported with their denominators, because a rate without its
    denominator is not a measurement.
    """
    clean, damaged = [], []
    for img, gts in gts_by_image.items():
        (damaged if gts else clean).append(img)

    def fires(img: str) -> bool:
        return any(p.score >= conf for p in preds_by_image.get(img, []))

    false_alarms = sum(1 for i in clean if fires(i))
    detected = sum(1 for i in damaged if fires(i))

    return {
        "conf": conf,
        "clean_frames": len(clean),
        "damaged_frames": len(damaged),
        "false_alarm_frames": false_alarms,
        "detected_frames": detected,
        "false_alarm_rate": (false_alarms / len(clean)) if clean else None,
        "recall": (detected / len(damaged)) if damaged else None,
        "precision": (detected / (detected + false_alarms)) if (detected + false_alarms) else None,
    }


def sweep(preds_by_image: dict[str, list[BBox]],
          gts_by_image: dict[str, list[BBox]],
          thresholds: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """Frame-level behaviour across a range of thresholds."""
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 19)]     # 0.05 .. 0.90
    return [measure(preds_by_image, gts_by_image, c) for c in thresholds]


def choose_threshold(preds_by_image: dict[str, list[BBox]],
                     gts_by_image: dict[str, list[BBox]],
                     target_false_alarm_rate: float = 0.05,
                     thresholds: Sequence[float] | None = None) -> dict[str, Any]:
    """Lowest threshold whose false-alarm rate meets the target.

    Lowest, not best-F1: subject to the false-alarm budget the operator will
    tolerate, the most useful model is the one that misses least. Picking by F1
    instead silently trades away the constraint that decides adoption.

    Returns the choice *and* the sweep, so the trade-off is visible rather than
    reduced to one number.
    """
    rows = sweep(preds_by_image, gts_by_image, thresholds)
    feasible = [r for r in rows
                if r["false_alarm_rate"] is not None
                and r["false_alarm_rate"] <= target_false_alarm_rate]
    chosen = min(feasible, key=lambda r: r["conf"]) if feasible else None
    return {
        "target_false_alarm_rate": target_false_alarm_rate,
        "chosen": chosen,
        "achievable": chosen is not None,
        "sweep": rows,
        "note": ("Calibrated on the supplied split only. A threshold does not "
                 "transfer between sites, cameras or water conditions — recalibrate "
                 "per deployment, and re-check after any change to either."
                 if chosen else
                 "No threshold in the sweep met the false-alarm target. Either the "
                 "budget is tighter than this model can serve, or the evaluation "
                 "set contains no clean frames to measure it on."),
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def gate(preds_by_image: dict[str, list[BBox]],
         gts_by_image: dict[str, list[BBox]],
         op: OperatingPoint | None = None) -> Verdict:
    """Decide whether this model meets the contract. Fails closed."""
    op = op or OperatingPoint()
    m = measure(preds_by_image, gts_by_image, op.conf)
    checks: list[Check] = []

    def add(name, value, bound, rule, detail):
        # An unmeasurable quantity is a failure, never a pass. Treating "no data"
        # as "no problem" is how an unvalidated model reaches a boat.
        if value is None:
            ok = False
        else:
            ok = value <= bound if rule == "<=" else value >= bound
        checks.append(Check(name, value, bound, rule, ok, detail))

    add("clean frames", m["clean_frames"], op.min_clean_frames, ">=",
        "frames of undamaged net available to measure false alarms on")
    add("damaged frames", m["damaged_frames"], op.min_damaged_frames, ">=",
        "labelled damage available to measure recall on")
    add("false alarm rate", m["false_alarm_rate"], op.max_false_alarm_rate, "<=",
        f"{m['false_alarm_frames']} of {m['clean_frames']} clean frames raised an alert")
    add("recall", m["recall"], op.min_recall, ">=",
        f"{m['detected_frames']} of {m['damaged_frames']} damaged frames were caught")
    if op.min_precision is not None:
        add("precision", m["precision"], op.min_precision, ">=",
            "share of alerting frames that really contained damage")

    return Verdict(passed=all(c.passed for c in checks), checks=checks,
                   measured=m, operating_point=op.to_dict())


def write_verdict(verdict: Verdict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
    return path


__all__ = ["OperatingPoint", "Check", "Verdict", "measure", "sweep",
           "choose_threshold", "gate", "write_verdict"]

"""The evidence ledger: what this project can and cannot claim.

Every number in this repo comes with a boundary. Damage is synthetic, composited
onto real undamaged net, so *false-alarm* behaviour is measured on real imagery
while *recall on real damage* has never been measured at all. Four clips from
one site over two days is a small sample of scenes, and frames within a clip are
correlated — so per-frame confidence intervals overstate what the data supports.

This module makes those boundaries machine-readable. Each :class:`Claim` records
a statement, the artifact backing it, and — critically — its
:class:`EvidenceLevel`. The assistant in :mod:`netinspect.assistant.agent` is
grounded in this ledger, which is what stops it answering "how accurate is it on
real damage?" with a number it has no right to.

The ledger is deliberately hand-curated rather than derived. A generated summary
would inherit whatever framing produced it; the point here is to state the
limits explicitly, including the ones no metric would surface on its own.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils import get_logger

LOGGER = get_logger()

RESULTS_DIR = Path("reports/results")


class EvidenceLevel(str, Enum):
    """How much weight a claim can carry.

    ``MEASURED_REAL``
        Measured on real imagery. The strongest level available here, and it
        still only ever covers false alarms — the real frames contain no damage.
    ``MEASURED_PROXY``
        Measured against synthetic damage composited onto real backgrounds.
        Informative about the pipeline; not evidence of real-world detection.
    ``INFERRED``
        A reading of measured results rather than a measurement itself.
    ``UNVALIDATED``
        Explicitly not measured. Answering as if it were is the failure mode
        this ledger exists to prevent.
    """
    MEASURED_REAL = "measured_on_real_data"
    MEASURED_PROXY = "measured_on_synthetic_proxy"
    INFERRED = "inferred_from_measurements"
    UNVALIDATED = "not_validated"

    @property
    def can_support_operational_decision(self) -> bool:
        return self in (EvidenceLevel.MEASURED_REAL, EvidenceLevel.MEASURED_PROXY)


@dataclass
class Claim:
    """One statement this project is or is not entitled to make."""
    topic: str
    statement: str
    level: EvidenceLevel
    artifact: str | None = None
    caveat: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        d["can_support_operational_decision"] = self.level.can_support_operational_decision
        return d


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
LEDGER: tuple[Claim, ...] = (
    Claim(
        topic="false_alarm_rate",
        statement=(
            "False-alarm behaviour is measured on 638 real frames of undamaged net "
            "across 4 clips and 2 days. Because the net is undamaged, every "
            "detection is a known false positive and no annotation was required."),
        level=EvidenceLevel.MEASURED_REAL,
        artifact="reports/results/operating_envelope/operating_envelope.json",
    ),
    Claim(
        topic="recall_on_real_damage",
        statement=(
            "NOT MEASURED. No real damaged net has ever been evaluated. Every "
            "recall, precision and F1 number in this repo is against synthetic "
            "damage composited onto real backgrounds by a single generator."),
        level=EvidenceLevel.UNVALIDATED,
        caveat=(
            "Any question about how well this finds real holes or tears must be "
            "answered with this limitation, not with a proxy number. Real labelled "
            "damage is the missing ingredient and no amount of further modelling "
            "substitutes for it."),
    ),
    Claim(
        topic="proxy_detection_performance",
        statement=(
            "On synthetic damage composited onto real frames, the model family "
            "spans F1 0.12 (label-free anomaly) to 0.97 (supervised segmentation). "
            "Train and test damage come from the same generator, so this measures "
            "the pipeline, not real-world detection."),
        level=EvidenceLevel.MEASURED_PROXY,
        artifact="reports/results/comparison_4way/",
        caveat="In-distribution by construction: one damage generator, train and test.",
    ),
    Claim(
        topic="scene_dominates_false_alarms",
        statement=(
            "False-alarm rate is dominated by which clip a frame came from, not by "
            "recording day or flight profile. Three training-day clips flown at "
            "near-identical standoff (0.60/0.66/0.71 m) gave detector false-alarm "
            "rates of 0.00, 0.33 and 0.00."),
        level=EvidenceLevel.MEASURED_REAL,
        artifact="reports/results/operating_envelope/operating_envelope.json",
    ),
    Claim(
        topic="standoff_hypothesis",
        statement=(
            "REJECTED. The hypothesis that the different-day gap is an "
            "operating-envelope violation driven by standoff distance is not "
            "supported: the clip spanning the widest standoff range (0.19-1.31 m) "
            "produces zero false alarms, and standoff shows no within-clip "
            "association on the held-out day (all p > 0.67)."),
        level=EvidenceLevel.MEASURED_REAL,
        artifact="reports/results/operating_envelope/operating_envelope.json",
        caveat="Reported as a negative result on purpose; the hypothesis was pre-registered.",
    ),
    Claim(
        topic="capture_quality_mechanism",
        statement=(
            "Capture quality is the strongest per-frame correlate of false alarms, "
            "and its sign depends on the model: the detector fires on sharp, "
            "high-contrast frames (r=+0.60) while the higher-capacity segmenter "
            "fires on degraded ones (r=-0.11)."),
        level=EvidenceLevel.MEASURED_REAL,
        artifact="reports/results/operating_envelope/operating_envelope.json",
    ),
    Claim(
        topic="what_the_detector_fires_on",
        statement=(
            "In the clip with the highest false-alarm rate, the detector's boxes "
            "land on the thin bright mooring cords rigged around fiducial "
            "calibration markers — not on the markers, and not on the net mesh. "
            "Elongated high-contrast structures resemble the synthetic tears it "
            "was trained on. A clean clip containing hardware further away fires "
            "nothing, so the trigger is a shape at a scale, not 'equipment in "
            "frame'."),
        level=EvidenceLevel.INFERRED,
        artifact="docs/images/what_the_detector_fires_on.jpg",
        caveat=(
            "An observation from the imagery of one clip, not a controlled "
            "experiment. It explains the measured sharpness correlation and is "
            "consistent with every clip-level rate, but 'cord present' was never "
            "quantified as a variable — that would need the frames labelled for "
            "rigging, which this project does not have."),
    ),
    Claim(
        topic="ensemble_mechanism",
        statement=(
            "The det-and-seg agreement ensemble suppresses false alarms because the "
            "two models fail in opposite capture-quality regimes, so requiring "
            "agreement cancels both failure modes."),
        level=EvidenceLevel.INFERRED,
        artifact="reports/results/ensemble/",
        caveat="A mechanism for an already-measured result, not a new accuracy claim.",
    ),
    Claim(
        topic="statistical_confidence",
        statement=(
            "Frames within a clip are strongly correlated (ICC up to 0.31 for the "
            "detector). The effective sample size is roughly 13 clips-worth of "
            "information, not 638 independent frames, so clip-level clustered "
            "intervals are far wider than naive per-frame intervals."),
        level=EvidenceLevel.MEASURED_REAL,
        artifact="reports/results/operating_envelope/operating_envelope.json",
        caveat=(
            "Any per-frame confidence interval quoted from this project overstates "
            "confidence in how it would generalise to a new net or a new site."),
    ),
    Claim(
        topic="cross_site_generalisation",
        statement=(
            "NOT MEASURED. All data comes from a single SINTEF SOLAQUA site over "
            "two days in August 2024. Nothing here establishes behaviour at another "
            "site, another camera, another net material, or another season."),
        level=EvidenceLevel.UNVALIDATED,
    ),
    Claim(
        topic="telemetry_coverage",
        statement=(
            "ROV telemetry (net-plane standoff, DVL velocity, depth, temperature, "
            "attitude, thruster effort) is extracted from all 5 SOLAQUA sensor bags "
            "and joined to frames on the bag clock, with 100% of frames timestamped "
            "and ~94% matched to a telemetry sample within 0.5 s."),
        level=EvidenceLevel.MEASURED_REAL,
        artifact="data/processed/telemetry/",
    ),
    Claim(
        topic="ssl_pretraining",
        statement=(
            "Domain SSL pretraining (SimCLR on 508 unlabelled frames) UNDERPERFORMS "
            "ImageNet transfer and off-the-shelf DINOv2. Reported as a negative "
            "result: SSL needs scale that 508 frames cannot provide."),
        level=EvidenceLevel.MEASURED_PROXY,
        artifact="reports/results/ssl_dino/",
    ),
    Claim(
        topic="augmentation_experiment",
        statement=(
            "Strong photometric augmentation (seg v4) FAILED to close the "
            "different-day gap (22% vs 18% baseline). Kept in the repo on purpose."),
        level=EvidenceLevel.MEASURED_PROXY,
        artifact="reports/results/adversarial_seg_v4/",
    ),
)

CLAIMS_BY_TOPIC: dict[str, Claim] = {c.topic: c for c in LEDGER}


# Question patterns that reach for an unvalidated claim. Kept as substrings
# rather than a model call so the guard is deterministic and testable offline.
#
# Two families, matching the two UNVALIDATED ledger entries: performance on real
# damage, and generalisation beyond the single site and two days in the data.
REAL_DAMAGE_TRIGGERS = (
    # recall_on_real_damage
    "real damage", "real hole", "real tear", "actual damage", "in production",
    "in the field", "deploy", "trust it", "accuracy on real", "catch a hole",
    "miss a hole", "find damage", "real net damage", "would it detect",
    "real net", "real fish farm",
    # cross_site_generalisation
    "different site", "another site", "different farm", "another farm",
    "our nets", "our site", "our farm", "different fish farm", "other sites",
    "different camera", "different rov", "different season", "new site",
)


def mentions_unvalidated_capability(question: str) -> bool:
    """True when a question reaches for the project's unvalidated claim.

    Used by the eval harness to check that the assistant surfaces the
    synthetic-proxy boundary rather than answering with a proxy number.
    """
    q = question.lower()
    return any(t in q for t in REAL_DAMAGE_TRIGGERS)


def ledger_dicts(level: EvidenceLevel | None = None) -> list[dict[str, Any]]:
    """Serialise the ledger, optionally filtered to one evidence level."""
    claims = LEDGER if level is None else [c for c in LEDGER if c.level == level]
    return [c.to_dict() for c in claims]


def unvalidated_topics() -> list[str]:
    """Topics this project explicitly cannot speak to."""
    return [c.topic for c in LEDGER if c.level == EvidenceLevel.UNVALIDATED]


def available_artifacts(root: str | Path = ".") -> dict[str, bool]:
    """Which artifacts referenced by the ledger actually exist on disk."""
    root = Path(root)
    return {c.artifact: (root / c.artifact).exists()
            for c in LEDGER if c.artifact}


def render_for_prompt() -> str:
    """Render the ledger as the grounding block of the assistant's system prompt."""
    lines = ["EVIDENCE LEDGER — the boundary of what this project may claim.", ""]
    for level in (EvidenceLevel.UNVALIDATED, EvidenceLevel.MEASURED_REAL,
                  EvidenceLevel.MEASURED_PROXY, EvidenceLevel.INFERRED):
        claims = [c for c in LEDGER if c.level == level]
        if not claims:
            continue
        lines.append(f"## {level.value}")
        for c in claims:
            lines.append(f"- [{c.topic}] {c.statement}")
            if c.artifact:
                lines.append(f"    artifact: {c.artifact}")
            if c.caveat:
                lines.append(f"    caveat: {c.caveat}")
        lines.append("")
    return "\n".join(lines)


def load_result(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON result artifact, returning None when it is absent."""
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Artifact not found: %s", p)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Artifact %s is not valid JSON: %s", p, exc)
        return None


__all__ = [
    "EvidenceLevel", "Claim", "LEDGER", "CLAIMS_BY_TOPIC", "REAL_DAMAGE_TRIGGERS",
    "mentions_unvalidated_capability", "ledger_dicts", "unvalidated_topics",
    "available_artifacts", "render_for_prompt", "load_result", "RESULTS_DIR",
]

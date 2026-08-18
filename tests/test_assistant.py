"""Tests for the evidence ledger and the assistant's grounding guard.

These run without an API key and without the anthropic SDK: the guard that
matters — refusing to answer past the evidence — is deterministic code, not a
model behaviour, precisely so it can be tested here.
"""
from __future__ import annotations

import pytest

from netinspect.assistant.evidence import (
    LEDGER,
    Claim,
    EvidenceLevel,
    available_artifacts,
    ledger_dicts,
    mentions_unvalidated_capability,
    render_for_prompt,
    unvalidated_topics,
)


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
def test_ledger_declares_real_damage_recall_unvalidated():
    """The project's central limitation must be in the ledger, explicitly."""
    assert "recall_on_real_damage" in unvalidated_topics()


def test_unvalidated_claims_cannot_support_decisions():
    for claim in LEDGER:
        if claim.level is EvidenceLevel.UNVALIDATED:
            assert not claim.level.can_support_operational_decision


def test_measured_claims_can_support_decisions():
    assert EvidenceLevel.MEASURED_REAL.can_support_operational_decision
    assert EvidenceLevel.MEASURED_PROXY.can_support_operational_decision
    assert not EvidenceLevel.INFERRED.can_support_operational_decision


def test_every_measured_claim_names_an_artifact():
    """A measured claim without a source is an unfalsifiable claim."""
    for claim in LEDGER:
        if claim.level in (EvidenceLevel.MEASURED_REAL, EvidenceLevel.MEASURED_PROXY):
            assert claim.artifact, f"{claim.topic} is measured but cites no artifact"


def test_topics_are_unique():
    topics = [c.topic for c in LEDGER]
    assert len(topics) == len(set(topics))


def test_ledger_serialises_with_decision_flag():
    dicts = ledger_dicts()
    assert len(dicts) == len(LEDGER)
    assert all("can_support_operational_decision" in d for d in dicts)
    assert all(isinstance(d["level"], str) for d in dicts)


def test_ledger_filters_by_level():
    unvalidated = ledger_dicts(EvidenceLevel.UNVALIDATED)
    assert unvalidated
    assert all(d["level"] == "not_validated" for d in unvalidated)


def test_prompt_block_leads_with_the_unvalidated_section():
    """What the model must not claim should be the first thing it reads."""
    rendered = render_for_prompt()
    assert rendered.index("not_validated") < rendered.index("measured_on_real_data")
    assert "recall_on_real_damage" in rendered


def test_prompt_block_includes_caveats():
    assert "real labelled damage" in render_for_prompt().lower()


def test_available_artifacts_reports_presence(tmp_path):
    result = available_artifacts(tmp_path)
    assert result and all(v is False for v in result.values())


def test_claim_roundtrips():
    c = Claim(topic="t", statement="s", level=EvidenceLevel.INFERRED, artifact="a.json")
    d = c.to_dict()
    assert d["level"] == "inferred_from_measurements"
    assert d["can_support_operational_decision"] is False


# --------------------------------------------------------------------------- #
# Trigger detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "How accurate is this on real damage?",
    "Can I deploy this to catch holes?",
    "Would it detect a real tear in the net?",
    "Can I trust it in the field?",
    "What is the accuracy on real nets?",
])
def test_real_damage_questions_are_flagged(question):
    assert mentions_unvalidated_capability(question)


@pytest.mark.parametrize("question", [
    "Which clip had the most false alarms?",
    "What was the water temperature?",
    "How many frames were analysed?",
    "What standoff distance was commanded?",
])
def test_operational_questions_are_not_flagged(question):
    assert not mentions_unvalidated_capability(question)


def test_trigger_detection_is_case_insensitive():
    assert mentions_unvalidated_capability("REAL DAMAGE performance?")


# --------------------------------------------------------------------------- #
# The grounding guard
# --------------------------------------------------------------------------- #
def _guard():
    from netinspect.assistant.agent import check_grounding
    return check_grounding


def test_proxy_number_without_caveat_is_flagged():
    """The failure this whole design exists to prevent."""
    g = _guard()(
        "How accurate is this at finding real damage?",
        "It achieves an F1 score of 0.97 with high precision.",
        ["get_false_alarm_analysis"],
    )
    assert g["question_touches_unvalidated_capability"]
    assert not g["answer_states_boundary"]
    assert g["missing_caveat"]


def test_stating_the_boundary_passes():
    g = _guard()(
        "How accurate is this at finding real damage?",
        "Recall on real damage has never been measured. Every figure in this "
        "project uses synthetic damage composited onto real backgrounds.",
        ["get_evidence"],
    )
    assert g["answer_states_boundary"]
    assert not g["missing_caveat"]


def test_operational_answer_needs_no_caveat():
    g = _guard()(
        "Which clip had the most false alarms?",
        "Clip 2024-08-22_14-29-05, at 33% of frames.",
        ["get_false_alarm_analysis"],
    )
    assert not g["question_touches_unvalidated_capability"]
    assert not g["missing_caveat"]


def test_guard_records_tool_usage():
    g = _guard()("Which clip?", "Clip A.", [])
    assert g["used_tools"] is False
    assert g["tool_count"] == 0

    g = _guard()("Which clip?", "Clip A.", ["a", "b"])
    assert g["used_tools"] is True
    assert g["tool_count"] == 2


@pytest.mark.parametrize("phrasing", [
    "all damage figures come from a synthetic generator",
    "this has not been measured on real nets",
    "performance on real damage is unvalidated",
    "the frames show undamaged net only",
])
def test_caveat_markers_are_recognised(phrasing):
    g = _guard()("How accurate on real damage?", phrasing, ["get_evidence"])
    assert g["answer_states_boundary"], phrasing


def test_answer_is_grounded_only_with_tools_and_caveat():
    from netinspect.assistant.agent import Answer

    good = Answer(text="Synthetic only.", question="real damage accuracy?",
                  tools_used=["get_evidence"],
                  grounding={"missing_caveat": False})
    assert good.is_grounded

    no_tools = Answer(text="Synthetic only.", question="q", tools_used=[],
                      grounding={"missing_caveat": False})
    assert not no_tools.is_grounded

    no_caveat = Answer(text="F1 0.97.", question="real damage accuracy?",
                       tools_used=["get_evidence"],
                       grounding={"missing_caveat": True})
    assert not no_caveat.is_grounded


# --------------------------------------------------------------------------- #
# The eval suite is itself worth checking
# --------------------------------------------------------------------------- #
def test_eval_suite_has_enough_adversarial_cases():
    from netinspect.assistant.eval_suite import SUITE

    caveat_cases = [c for c in SUITE if c.must_caveat]
    assert len(caveat_cases) >= 4, "too few adversarial cases on the core limitation"


def test_every_caveat_case_asks_a_question_the_guard_flags():
    """A must_caveat case the guard does not flag would pass vacuously.

    This is the test that keeps the eval honest: without it, adding a
    politely-phrased question to the suite would silently inflate the
    boundary-disclosure rate without measuring anything.
    """
    from netinspect.assistant.eval_suite import SUITE

    unflagged = [c.id for c in SUITE
                 if c.must_caveat and not mentions_unvalidated_capability(c.question)]
    assert not unflagged, f"must_caveat cases the guard misses: {unflagged}"


def test_eval_suite_ids_are_unique():
    from netinspect.assistant.eval_suite import SUITE

    assert len({c.id for c in SUITE}) == len(SUITE)


def test_non_caveat_cases_are_not_accidentally_flagged():
    """Grounded-retrieval cases must not trip the boundary guard."""
    from netinspect.assistant.eval_suite import SUITE

    for case in SUITE:
        if not case.must_caveat:
            assert not mentions_unvalidated_capability(case.question), \
                f"{case.id} trips the guard but does not declare must_caveat"

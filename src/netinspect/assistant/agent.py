"""An inspection assistant that knows the limits of its own evidence.

Most "chat with your data" demos will happily answer a question the underlying
data cannot support. For a net-inspection system that failure is not cosmetic:
"how well does it detect damage?" has never been measured on real damage here,
and an assistant that answers it with the synthetic-proxy F1 would be actively
misleading an operator about escape risk.

So the assistant is built the other way round. Its system prompt is generated
from :mod:`netinspect.assistant.evidence` — the machine-readable ledger of what
this project may and may not claim — and every tool it can call returns the
artifact path backing its numbers. The design goal is not fluency; it is that
each answer be traceable and each limit be stated without being asked.

Two guards operate at different layers, deliberately:

* **Model-level** — the ledger in the system prompt, plus per-tool
  ``interpretation_required`` fields that travel with the data.
* **Code-level** — :func:`check_grounding`, a deterministic post-check that
  flags an answer touching unvalidated capability without the caveat. It is
  plain string matching, testable without an API key, and does not depend on
  the model having behaved.

Usage::

    from netinspect.assistant import InspectionAssistant
    answer = InspectionAssistant().ask("Which clip produced the most false alarms?")
    print(answer.text)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils import get_logger
from .backends import Backend, make_backend
from .evidence import mentions_unvalidated_capability, render_for_prompt

LOGGER = get_logger()

DEFAULT_BACKEND = "anthropic"

# Phrases that count as surfacing the synthetic-damage boundary.
CAVEAT_MARKERS = (
    "synthetic", "not measured", "not been measured", "unvalidated", "not validated",
    "no real damage", "undamaged", "proxy", "cannot claim", "has not been evaluated",
    "real labelled damage", "real labeled damage",
)

SYSTEM_PROMPT = """\
You are the inspection assistant for an aquaculture net-inspection computer-vision
project. You answer operational questions about inspection runs, ROV telemetry, and
model behaviour, for engineers and operations staff.

Your defining constraint: **you may only state what the evidence supports, and you
must say so when it does not.** This project's credibility rests on not overclaiming,
and you are part of that guarantee.

Rules, in priority order:

1. Ground every factual claim in a tool call. Do not answer numeric questions from
   memory or from this prompt — call a tool and cite the `artifact` path it returns.
2. If a question reaches for something in the ledger marked `not_validated`, say so
   plainly and first. The most common case: recall on REAL damage has never been
   measured. Every recall/precision/F1 figure here is against synthetic damage from a
   single generator composited onto real backgrounds. Never present a proxy number as
   an answer to "how well would this find real damage?" — state the limitation, then
   offer what IS measured (false-alarm behaviour on real undamaged net).
3. Prefer clip-clustered confidence intervals over naive per-frame ones, and say why:
   frames within a clip are correlated, so the effective sample size is far below the
   frame count.
4. When a tool result carries an `interpretation_required` field, honour it.
5. If a tool returns an error or missing artifact, say the data is not available. Do
   not substitute a plausible number.
6. Be direct and brief. Lead with the answer, then the evidence, then the caveat.
   Do not hedge on things that ARE measured — state those plainly.

You are not being cautious for its own sake. Measured false-alarm behaviour on real
imagery is real evidence and should be stated with confidence. The discipline is
about the boundary between that and what has not been measured.

{ledger}
"""


@dataclass
class Answer:
    """One assistant response plus the trace needed to audit it."""
    text: str
    question: str
    tools_used: list[str] = field(default_factory=list)
    artifacts_cited: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    grounding: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    backend: str = ""
    model: str = ""

    @property
    def is_grounded(self) -> bool:
        """True when the answer cited a tool and met the caveat requirement."""
        return bool(self.tools_used) and not self.grounding.get("missing_caveat", False)


def check_grounding(question: str, answer_text: str,
                    tools_used: list[str]) -> dict[str, Any]:
    """Deterministic post-check on one answer.

    Deliberately simple string matching rather than a model call: this is the
    layer that must hold even when the model does not, so it has to be
    inspectable and testable offline, and identical across backends. It detects
    the failure that matters — answering an unvalidated-capability question
    without stating the boundary — and does not attempt to judge answer quality.
    """
    lowered = answer_text.lower()
    touches_unvalidated = mentions_unvalidated_capability(question)
    has_caveat = any(m in lowered for m in CAVEAT_MARKERS)
    return {
        "question_touches_unvalidated_capability": touches_unvalidated,
        "answer_states_boundary": has_caveat,
        "missing_caveat": touches_unvalidated and not has_caveat,
        "used_tools": bool(tools_used),
        "tool_count": len(tools_used),
    }


def extract_artifacts(text: str) -> list[str]:
    """Pull artifact paths the answer cited, so citations can be checked."""
    found: list[str] = []
    for token in text.split():
        cleaned = token.strip("`,.()[]'\"")
        if any(m in cleaned for m in ("reports/results/", "data/processed/")) \
                and cleaned not in found:
            found.append(cleaned)
    return found


class InspectionAssistant:
    """Tool-calling assistant grounded in the project's evidence ledger.

    The backend is swappable so the same guardrail can be measured across
    models — see :mod:`netinspect.assistant.backends`. Everything that enforces
    grounding (the ledger in the system prompt, artifact-cited tool results, the
    post-check) is backend-independent by construction, so a difference in
    results is a difference in the model, not in the harness.

    Parameters
    ----------
    backend : str
        ``"anthropic"`` or ``"ollama"``.
    model : str, optional
        Model id; defaults to the backend's default.
    **kwargs
        Passed to the backend (``effort``, ``max_tokens`` for Anthropic;
        ``host``, ``temperature``, ``num_ctx`` for Ollama).
    """

    def __init__(self, backend: str | Backend = DEFAULT_BACKEND,
                 model: str | None = None, **kwargs):
        self.backend: Backend = (backend if not isinstance(backend, str)
                                 else make_backend(backend, model, **kwargs))
        self.system_prompt = SYSTEM_PROMPT.format(ledger=render_for_prompt())

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        """Answer one question, running the tool loop to completion."""
        turn = self.backend.run(self.system_prompt, question, history)
        grounding = check_grounding(question, turn.text, turn.tools_used)
        if grounding["missing_caveat"]:
            LOGGER.warning("Answer touched unvalidated capability without a caveat.")
        return Answer(
            text=turn.text, question=question, tools_used=turn.tools_used,
            artifacts_cited=extract_artifacts(turn.text),
            stop_reason=turn.stop_reason, grounding=grounding, usage=turn.usage,
            backend=self.backend.name, model=self.backend.model,
        )


__all__ = ["InspectionAssistant", "Answer", "check_grounding", "extract_artifacts",
           "SYSTEM_PROMPT", "CAVEAT_MARKERS", "DEFAULT_BACKEND"]

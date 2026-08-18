"""Grounded inspection assistant: tool-calling Q&A over this repo's real artifacts.

See :mod:`netinspect.assistant.agent` for the design rationale — the system
prompt is generated from the project's evidence ledger, so the assistant states
the boundary of what has been measured instead of answering past it. The model
backend is swappable (:mod:`netinspect.assistant.backends`) so that guardrail
can be measured across models rather than assumed.
"""
from .agent import Answer, InspectionAssistant, check_grounding, extract_artifacts
from .backends import AnthropicBackend, OllamaBackend, Turn, make_backend
from .eval_suite import SUITE, Case
from .evidence import (
    LEDGER,
    Claim,
    EvidenceLevel,
    ledger_dicts,
    mentions_unvalidated_capability,
    render_for_prompt,
    unvalidated_topics,
)
from .tools import TOOL_NAMES, TOOL_SPECS, anthropic_tools, call_tool, ollama_tools

__all__ = [
    "InspectionAssistant", "Answer", "check_grounding", "extract_artifacts",
    "make_backend", "AnthropicBackend", "OllamaBackend", "Turn",
    "LEDGER", "Claim", "EvidenceLevel", "ledger_dicts", "render_for_prompt",
    "unvalidated_topics", "mentions_unvalidated_capability",
    "TOOL_SPECS", "TOOL_NAMES", "anthropic_tools", "ollama_tools", "call_tool",
    "SUITE", "Case",
]

"""Tests for the swappable model backends and the provider-neutral tool specs.

No network and no model: these cover the parts that must be identical across
backends. That equivalence is what makes a cross-model comparison meaningful —
if the Anthropic and Ollama paths exposed different tool schemas or ran
different loops, a difference in eval results would say more about the harness
than about the models.
"""
from __future__ import annotations

import json

import pytest

from netinspect.assistant.backends import (
    MAX_TOOL_ROUNDS,
    OllamaBackend,
    Turn,
    make_backend,
)
from netinspect.assistant.tools import (
    SPECS_BY_NAME,
    TOOL_NAMES,
    TOOL_SPECS,
    anthropic_tools,
    call_tool,
    ollama_tools,
)


# --------------------------------------------------------------------------- #
# Tool specs are shared, so both providers see the same contract
# --------------------------------------------------------------------------- #
def test_both_providers_expose_the_same_tools():
    assert {t["name"] for t in anthropic_tools()} == set(TOOL_NAMES)
    assert {t["function"]["name"] for t in ollama_tools()} == set(TOOL_NAMES)


def test_both_providers_expose_identical_schemas():
    """Schema drift between providers would masquerade as a model difference."""
    anth = {t["name"]: t["input_schema"] for t in anthropic_tools()}
    olla = {t["function"]["name"]: t["function"]["parameters"] for t in ollama_tools()}
    assert anth == olla


def test_both_providers_expose_identical_descriptions():
    anth = {t["name"]: t["description"] for t in anthropic_tools()}
    olla = {t["function"]["name"]: t["function"]["description"] for t in ollama_tools()}
    assert anth == olla


def test_schemas_are_valid_json_schema_objects():
    for tool in anthropic_tools():
        s = tool["input_schema"]
        assert s["type"] == "object"
        assert s["additionalProperties"] is False
        assert isinstance(s["properties"], dict)
        assert set(s["required"]) <= set(s["properties"])


def test_optional_parameters_are_not_required():
    schema = SPECS_BY_NAME["query_telemetry"].json_schema()
    assert "statistic" in schema["properties"]
    assert "statistic" not in schema["required"]
    assert "clip" in schema["required"]


def test_required_flag_is_stripped_from_emitted_properties():
    """`required` is our bookkeeping, not part of the property schema."""
    for tool in ollama_tools():
        for prop in tool["function"]["parameters"]["properties"].values():
            assert "required" not in prop


def test_zero_argument_tools_have_empty_properties():
    schema = SPECS_BY_NAME["list_inspections"].json_schema()
    assert schema["properties"] == {}
    assert schema["required"] == []


def test_every_spec_has_a_description_and_callable():
    for spec in TOOL_SPECS:
        assert spec.description.strip()
        assert callable(spec.fn)


# --------------------------------------------------------------------------- #
# call_tool is the single execution path for both backends
# --------------------------------------------------------------------------- #
def test_unknown_tool_returns_an_error_payload_not_an_exception():
    out = json.loads(call_tool("no_such_tool", {}))
    assert "error" in out
    assert "available_tools" in out


def test_bad_arguments_return_the_expected_schema():
    out = json.loads(call_tool("get_evidence", {"wrong_arg": 1}))
    assert "error" in out
    assert "expected_schema" in out


def test_evidence_tool_executes_and_returns_json():
    out = json.loads(call_tool("get_evidence", {"topic": "recall_on_real_damage"}))
    assert out["level"] == "not_validated"
    assert out["can_support_operational_decision"] is False


def test_evidence_tool_lists_topics():
    out = json.loads(call_tool("get_evidence", {"topic": "all"}))
    assert "recall_on_real_damage" in out["unvalidated"]


def test_missing_artifact_is_reported_not_invented():
    """A missing file must produce an error the model can see, never a guess."""
    out = json.loads(call_tool("query_telemetry", {
        "clip": "1999-01-01_00-00-00", "stream": "net_plane",
        "column": "net_distance", "statistic": "mean"}))
    assert "error" in out


# --------------------------------------------------------------------------- #
# Backend construction and argument handling
# --------------------------------------------------------------------------- #
def test_make_backend_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_backend("gpt5000")


def test_ollama_backend_constructs_without_a_daemon():
    """Construction must not perform I/O — only run() should touch the network."""
    b = make_backend("ollama", "qwen3:14b")
    assert b.name == "ollama"
    assert b.model == "qwen3:14b"


def test_ollama_backend_drops_anthropic_only_kwargs():
    """`effort` is meaningless locally and must not reach the Ollama client."""
    b = make_backend("ollama", "qwen3:14b", effort="high", max_tokens=1234)
    assert b.model == "qwen3:14b"


def test_ollama_parses_string_tool_arguments():
    """Older Ollama builds hand back arguments as a JSON string."""
    call = {"function": {"name": "get_evidence", "arguments": '{"topic": "all"}'}}
    assert OllamaBackend._arguments(call) == {"topic": "all"}


def test_ollama_parses_dict_tool_arguments():
    call = {"function": {"name": "get_evidence", "arguments": {"topic": "all"}}}
    assert OllamaBackend._arguments(call) == {"topic": "all"}


def test_ollama_survives_unparseable_tool_arguments():
    call = {"function": {"name": "get_evidence", "arguments": "{not json"}}
    assert OllamaBackend._arguments(call) == {}


def test_ollama_handles_missing_arguments_key():
    assert OllamaBackend._arguments({"function": {"name": "list_inspections"}}) == {}


def test_unreachable_ollama_gives_an_actionable_error():
    b = OllamaBackend(model="qwen3:14b", host="http://localhost:1", timeout=2)
    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        b.run("system", "question")


def test_tool_round_cap_is_bounded():
    """A model that loops on tool calls must terminate, not run forever."""
    assert 1 <= MAX_TOOL_ROUNDS <= 32


def test_turn_defaults_are_empty_not_none():
    t = Turn(text="hi")
    assert t.tools_used == [] and t.usage == {} and t.rounds == 0

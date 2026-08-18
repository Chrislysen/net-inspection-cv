"""Tests for the OpenAI-compatible assistant backend.

No network. A stubbed transport stands in for the endpoint, so the tool loop,
the message shapes and the error handling are all exercised offline.

The shape details tested here are the ones a naive copy of the Ollama backend
gets wrong: responses nested under `choices`, `usage` field names, and the
`tool_call_id` that strict servers require on every tool result.
"""
from __future__ import annotations

import json

import pytest

from netinspect.assistant import backends as B


def _backend(monkeypatch, responses):
    """A backend whose HTTP transport replays the given responses in order."""
    calls = []
    seq = list(responses)

    def fake_post(self, payload):
        calls.append(payload)
        return seq.pop(0)

    monkeypatch.setattr(B.OpenAICompatBackend, "_post", fake_post)
    b = B.OpenAICompatBackend(model="test-model", base_url="https://x/v1",
                              api_key="secret-key")
    return b, calls


def _msg(content=None, tool_calls=None, finish="stop", prompt=10, completion=5):
    return {
        "choices": [{"message": {"content": content, "tool_calls": tool_calls},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_a_missing_key_fails_with_an_actionable_message():
    with pytest.raises(RuntimeError) as e:
        B.OpenAICompatBackend(env={})
    msg = str(e.value)
    assert B.ENV_OPENAI_API_KEY in msg
    assert "nousresearch" in msg, "should name a concrete endpoint to point at"


def test_configuration_comes_from_the_environment():
    b = B.OpenAICompatBackend(env={
        B.ENV_OPENAI_API_KEY: "k",
        B.ENV_OPENAI_BASE_URL: "https://inference-api.nousresearch.com/v1",
        B.ENV_OPENAI_MODEL: "Hermes-4-70B",
    })
    assert b.base_url == "https://inference-api.nousresearch.com/v1"
    assert b.model == "Hermes-4-70B"


def test_a_trailing_slash_on_the_base_url_does_not_double_up():
    b = B.OpenAICompatBackend(base_url="https://x/v1/", api_key="k")
    assert b.base_url == "https://x/v1"


def test_named_providers_preselect_a_base_url_but_never_a_key():
    b = B.make_backend("nous", api_key="k", model="m")
    assert b.base_url == B.KNOWN_OPENAI_COMPATIBLE["nous"]
    with pytest.raises(RuntimeError):
        B.make_backend("nous", env={})          # still needs a key


def test_a_self_hosted_endpoint_is_just_a_base_url():
    """A vLLM on Modal, or anything else speaking this shape."""
    b = B.make_backend("modal", base_url="https://me--vllm.modal.run/v1",
                       api_key="k", model="mine")
    assert b.base_url == "https://me--vllm.modal.run/v1"


def test_an_unknown_backend_name_lists_the_alternatives():
    with pytest.raises(ValueError) as e:
        B.make_backend("nonsense")
    assert "nous" in str(e.value) and "ollama" in str(e.value)


def test_anthropic_only_arguments_are_dropped_rather_than_exploding():
    b = B.make_backend("openai", api_key="k", model="m",
                       effort="high", max_tokens=100)
    assert b.model == "m"


# --------------------------------------------------------------------------- #
# the tool loop
# --------------------------------------------------------------------------- #
def test_a_plain_answer_returns_without_calling_tools(monkeypatch):
    b, calls = _backend(monkeypatch, [_msg(content="42 frames.")])
    turn = b.run("sys", "how many?")
    assert turn.text == "42 frames."
    assert turn.tools_used == []
    assert turn.rounds == 1
    assert calls[0]["messages"][0]["role"] == "system"


def test_a_tool_call_is_executed_and_answered(monkeypatch):
    tool_name = B.ollama_tools()[0]["function"]["name"]
    monkeypatch.setattr(B, "call_tool", lambda name, args: "TOOL RESULT")
    b, calls = _backend(monkeypatch, [
        _msg(tool_calls=[{"id": "call_1", "type": "function",
                          "function": {"name": tool_name, "arguments": "{}"}}],
             finish="tool_calls"),
        _msg(content="Grounded answer."),
    ])
    turn = b.run("sys", "q")
    assert turn.text == "Grounded answer."
    assert turn.tools_used == [tool_name]
    assert turn.rounds == 2


def test_every_tool_result_carries_the_id_of_the_call_it_answers(monkeypatch):
    """Strict servers reject the conversation without tool_call_id."""
    tool_name = B.ollama_tools()[0]["function"]["name"]
    monkeypatch.setattr(B, "call_tool", lambda name, args: "R")
    b, calls = _backend(monkeypatch, [
        _msg(tool_calls=[{"id": "call_abc", "type": "function",
                          "function": {"name": tool_name, "arguments": "{}"}}],
             finish="tool_calls"),
        _msg(content="done"),
    ])
    b.run("sys", "q")
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_abc"


def test_string_and_dict_tool_arguments_are_both_accepted(monkeypatch):
    seen = {}
    tool_name = B.ollama_tools()[0]["function"]["name"]
    monkeypatch.setattr(B, "call_tool",
                        lambda name, args: seen.setdefault(name, args) or "R")
    b, _ = _backend(monkeypatch, [
        _msg(tool_calls=[{"id": "1", "function": {"name": tool_name,
                                                  "arguments": json.dumps({"a": 1})}}],
             finish="tool_calls"),
        _msg(content="x"),
    ])
    b.run("s", "q")
    assert seen[tool_name] == {"a": 1}


def test_unparseable_tool_arguments_degrade_to_empty_rather_than_crashing():
    assert B.OpenAICompatBackend._arguments(
        {"function": {"name": "x", "arguments": "{not json"}}) == {}


def test_token_usage_is_summed_across_rounds(monkeypatch):
    tool_name = B.ollama_tools()[0]["function"]["name"]
    monkeypatch.setattr(B, "call_tool", lambda name, args: "R")
    b, _ = _backend(monkeypatch, [
        _msg(tool_calls=[{"id": "1", "function": {"name": tool_name, "arguments": "{}"}}],
             finish="tool_calls", prompt=100, completion=20),
        _msg(content="done", prompt=150, completion=30),
    ])
    turn = b.run("s", "q")
    assert turn.usage == {"input_tokens": 250, "output_tokens": 50}


def test_the_tool_loop_is_capped(monkeypatch):
    """A model that calls tools forever must terminate, not spin."""
    tool_name = B.ollama_tools()[0]["function"]["name"]
    monkeypatch.setattr(B, "call_tool", lambda name, args: "R")
    forever = [_msg(tool_calls=[{"id": str(i), "function": {"name": tool_name,
                                                            "arguments": "{}"}}],
                    finish="tool_calls") for i in range(B.MAX_TOOL_ROUNDS + 3)]
    b, _ = _backend(monkeypatch, forever)
    turn = b.run("s", "q")
    assert turn.stop_reason == "max_rounds"
    assert turn.rounds == B.MAX_TOOL_ROUNDS


def test_an_empty_response_is_an_error_not_a_silent_blank(monkeypatch):
    b, _ = _backend(monkeypatch, [{"choices": [], "usage": {}}])
    with pytest.raises(RuntimeError, match="No choices"):
        b.run("s", "q")


# --------------------------------------------------------------------------- #
# secret handling
# --------------------------------------------------------------------------- #
def test_the_key_is_not_exposed_by_repr_or_str():
    """A backend printed into a log or a traceback must not leak the key."""
    b = B.OpenAICompatBackend(base_url="https://x/v1", api_key="SUPER-SECRET")
    assert "SUPER-SECRET" not in repr(b)
    assert "SUPER-SECRET" not in str(b)


def test_the_key_is_not_in_the_no_key_error_message():
    with pytest.raises(RuntimeError) as e:
        B.OpenAICompatBackend(env={B.ENV_OPENAI_BASE_URL: "https://x/v1"})
    assert "None" not in str(e.value).split("Set ")[0]

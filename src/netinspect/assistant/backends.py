"""Model backends for the inspection assistant: Anthropic API and local Ollama.

The assistant's value is its *guardrail* — it must state the boundary of what
this project has measured instead of answering past it. That guardrail is a
property of the system (evidence ledger in the prompt, artifact-cited tool
results, a deterministic post-check), not of any one model. Which raises the
question the eval harness exists to answer: **do different models actually
honour it?**

So the backend is swappable, and both implementations run the same manual tool
loop against the same provider-neutral tool specs in
:mod:`netinspect.assistant.tools`. A manual loop rather than the Anthropic
SDK's tool runner, because a runner-vs-hand-rolled-loop difference between
backends would show up in the results and be misread as a model difference.

Running the eval on a local model is not a cheaper substitute for running it on
a frontier model — it is a different measurement. Report them separately.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..utils import get_logger, optional_import
from .tools import anthropic_tools, call_tool, ollama_tools

LOGGER = get_logger()

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OLLAMA_MODEL = "qwen3:14b"
OLLAMA_HOST = "http://localhost:11434"
MAX_TOOL_ROUNDS = 8

# Any OpenAI-compatible endpoint. Configured from the environment only — a key
# passed as an argument lands in shell history and in `ps` output.
ENV_OPENAI_API_KEY = "NETINSPECT_OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "NETINSPECT_OPENAI_BASE_URL"
ENV_OPENAI_MODEL = "NETINSPECT_OPENAI_MODEL"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Endpoints this shape is known to cover, for the error messages and the docs.
KNOWN_OPENAI_COMPATIBLE = {
    "nous": "https://inference-api.nousresearch.com/v1",
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "local-vllm": "http://localhost:8000/v1",
}


@dataclass
class Turn:
    """The outcome of one full question, after the tool loop settles."""
    text: str
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    rounds: int = 0


class Backend(Protocol):
    """A model that can run a grounded, tool-using turn."""

    name: str
    model: str

    def run(self, system: str, question: str,
            history: list[dict] | None = None) -> Turn: ...


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    """Claude via the Messages API, with an explicit tool loop."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, effort: str = "medium",
                 max_tokens: int = 16000, client: Any = None):
        anthropic = optional_import("anthropic")
        if anthropic is None:
            raise RuntimeError("Install the anthropic SDK: pip install -e '.[assistant]'")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()
        self.tools = anthropic_tools()

    def run(self, system: str, question: str,
            history: list[dict] | None = None) -> Turn:
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": question})

        tools_used: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        text = ""
        stop_reason = None

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            resp = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                tools=self.tools, messages=messages,
            )
            stop_reason = resp.stop_reason
            if getattr(resp, "usage", None):
                usage["input_tokens"] += getattr(resp.usage, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(resp.usage, "output_tokens", 0) or 0

            if stop_reason == "refusal":
                return Turn("The model declined to answer this request.",
                            tools_used, stop_reason, usage, round_no)

            said = "".join(b.text for b in resp.content if b.type == "text")
            if said.strip():
                text = said

            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                return Turn(text, tools_used, stop_reason, usage, round_no)

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in calls:
                tools_used.append(block.name)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": call_tool(block.name, dict(block.input))})
            messages.append({"role": "user", "content": results})

        LOGGER.warning("Anthropic backend hit the %d-round tool cap", MAX_TOOL_ROUNDS)
        return Turn(text, tools_used, "max_rounds", usage, MAX_TOOL_ROUNDS)


# --------------------------------------------------------------------------- #
# Ollama (local)
# --------------------------------------------------------------------------- #
class OllamaBackend:
    """A local model served by Ollama, via its ``/api/chat`` tool-calling shape.

    Ollama accepts OpenAI-style function definitions and returns
    ``message.tool_calls``. Arguments arrive already parsed as a dict on recent
    versions, but older ones hand back a JSON string, so both are accepted.

    Not every local model supports tools. Ollama answers with an explicit
    "does not support tools" error, which is surfaced rather than silently
    degrading into an ungrounded chat — an assistant answering from memory is
    precisely the failure this project is built to avoid.
    """

    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str = OLLAMA_HOST,
                 timeout: int = 300, temperature: float = 0.0,
                 num_ctx: int = 16384):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.tools = ollama_tools()

    # -- transport ----------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if "does not support tools" in detail:
                raise RuntimeError(
                    f"Ollama model {self.model!r} does not support tool calling. "
                    "Try qwen3:14b or qwen2.5:14b-instruct.") from exc
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host} ({exc.reason}). "
                "Is the daemon running? Try: ollama serve") from exc

    def available_models(self) -> list[str]:
        req = urllib.request.Request(f"{self.host}/api/tags")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]

    @staticmethod
    def _arguments(call: dict) -> dict:
        args = (call.get("function") or {}).get("arguments", {})
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                LOGGER.warning("Ollama returned unparseable tool arguments: %r", args)
                return {}
        return args or {}

    # -- turn ---------------------------------------------------------------
    def run(self, system: str, question: str,
            history: list[dict] | None = None) -> Turn:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        tools_used: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        text = ""

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            data = self._post("/api/chat", {
                "model": self.model, "messages": messages, "tools": self.tools,
                "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            })
            usage["input_tokens"] += int(data.get("prompt_eval_count") or 0)
            usage["output_tokens"] += int(data.get("eval_count") or 0)

            msg = data.get("message") or {}
            said = (msg.get("content") or "").strip()
            if said:
                text = said

            calls = msg.get("tool_calls") or []
            if not calls:
                return Turn(text, tools_used, data.get("done_reason") or "stop",
                            usage, round_no)

            messages.append(msg)
            for call in calls:
                name = (call.get("function") or {}).get("name", "")
                tools_used.append(name)
                result = call_tool(name, self._arguments(call))
                messages.append({"role": "tool", "content": result, "name": name})

        LOGGER.warning("Ollama backend hit the %d-round tool cap", MAX_TOOL_ROUNDS)
        return Turn(text, tools_used, "max_rounds", usage, MAX_TOOL_ROUNDS)


class OpenAICompatBackend:
    """Any endpoint speaking the OpenAI ``/chat/completions`` shape.

    One implementation covers Nous Research, OpenAI, Together, Groq, Fireworks,
    and — the case that motivated it — a model you host yourself, since vLLM and
    TGI both expose this API. A vLLM served on Modal is just a base URL here.

    That matters for the measurement this module exists to support. The eval
    answers "do different models honour the guardrail?", and until now it could
    only ask two families. A self-hosted or third-party endpoint is a third
    independent answer, which is worth more than a second frontier model.

    Differences from the Ollama shape, all of them small and all of them places
    a naive copy would break: responses arrive under ``choices[0].message``,
    token counts are ``usage.prompt_tokens``/``completion_tokens``, and a tool
    result **must** carry the ``tool_call_id`` it answers — omit it and strict
    servers reject the whole conversation.

    The key is read from the environment and never logged. It is not accepted as
    a command-line argument, because arguments end up in shell history and
    process listings.
    """

    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: int = 300,
                 temperature: float = 0.0, env: dict[str, str] | None = None):
        import os

        env = os.environ if env is None else env
        self.base_url = (base_url or env.get(ENV_OPENAI_BASE_URL)
                         or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.model = model or env.get(ENV_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL
        self._key = api_key or env.get(ENV_OPENAI_API_KEY) or ""
        if not self._key:
            raise RuntimeError(
                f"No API key. Set {ENV_OPENAI_API_KEY} in your shell (not on the "
                f"command line), plus {ENV_OPENAI_BASE_URL} if you are not using "
                f"{DEFAULT_OPENAI_BASE_URL} — e.g. "
                "https://inference-api.nousresearch.com/v1 for Nous, or your own "
                "vLLM/Modal endpoint.")
        self.timeout = timeout
        self.temperature = temperature
        self.tools = ollama_tools()          # the OpenAI function-calling schema

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"HTTP {exc.code} from {self.base_url} — the key was rejected. "
                    f"Check {ENV_OPENAI_API_KEY} matches the endpoint.") from exc
            if "tool" in detail.lower() and exc.code == 400:
                raise RuntimeError(
                    f"{self.model!r} appears not to support tool calling on this "
                    f"endpoint. HTTP 400: {detail}") from exc
            raise RuntimeError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach {self.base_url} ({exc.reason}).") from exc

    @staticmethod
    def _arguments(call: dict) -> dict:
        args = (call.get("function") or {}).get("arguments", {})
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                LOGGER.warning("Unparseable tool arguments: %r", args)
                return {}
        return args or {}

    def run(self, system: str, question: str,
            history: list[dict] | None = None) -> Turn:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})

        tools_used: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        text = ""

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            data = self._post({
                "model": self.model, "messages": messages, "tools": self.tools,
                "temperature": self.temperature, "stream": False,
            })
            u = data.get("usage") or {}
            usage["input_tokens"] += int(u.get("prompt_tokens") or 0)
            usage["output_tokens"] += int(u.get("completion_tokens") or 0)

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices in response from {self.base_url}")
            choice = choices[0]
            msg = choice.get("message") or {}
            said = (msg.get("content") or "").strip()
            if said:
                text = said

            calls = msg.get("tool_calls") or []
            if not calls:
                return Turn(text, tools_used, choice.get("finish_reason") or "stop",
                            usage, round_no)

            messages.append(msg)
            for call in calls:
                name = (call.get("function") or {}).get("name", "")
                tools_used.append(name)
                result = call_tool(name, self._arguments(call))
                messages.append({
                    "role": "tool",
                    # Required by strict servers; the id ties the result to its call.
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": result,
                })

        LOGGER.warning("OpenAI-compatible backend hit the %d-round tool cap",
                       MAX_TOOL_ROUNDS)
        return Turn(text, tools_used, "max_rounds", usage, MAX_TOOL_ROUNDS)


def make_backend(kind: str = "anthropic", model: str | None = None,
                 **kwargs) -> Backend:
    """Construct a backend by name."""
    kind = kind.lower()
    if kind == "anthropic":
        return AnthropicBackend(model=model or DEFAULT_ANTHROPIC_MODEL, **kwargs)
    if kind == "ollama":
        kwargs.pop("effort", None)      # Anthropic-only knob
        kwargs.pop("max_tokens", None)
        return OllamaBackend(model=model or DEFAULT_OLLAMA_MODEL, **kwargs)
    if kind in ("openai", "openai-compat", "nous", "together", "groq",
                "vllm", "local-vllm", "modal"):
        kwargs.pop("effort", None)
        kwargs.pop("max_tokens", None)
        # A named provider only preselects the base URL; the key still comes
        # from the environment.
        if kind in KNOWN_OPENAI_COMPATIBLE and "base_url" not in kwargs:
            kwargs["base_url"] = KNOWN_OPENAI_COMPATIBLE[kind]
        return OpenAICompatBackend(model=model, **kwargs)
    raise ValueError(
        f"Unknown backend {kind!r}. Choose 'anthropic', 'ollama', or an "
        f"OpenAI-compatible endpoint: {', '.join(sorted(KNOWN_OPENAI_COMPATIBLE))}, "
        "'modal', 'vllm', 'openai-compat'.")


__all__ = ["Backend", "Turn", "AnthropicBackend", "OllamaBackend", "make_backend",
           "OpenAICompatBackend", "KNOWN_OPENAI_COMPATIBLE",
           "ENV_OPENAI_API_KEY", "ENV_OPENAI_BASE_URL", "ENV_OPENAI_MODEL",
           "DEFAULT_OPENAI_BASE_URL", "DEFAULT_OPENAI_MODEL",
           "DEFAULT_ANTHROPIC_MODEL", "DEFAULT_OLLAMA_MODEL", "OLLAMA_HOST",
           "MAX_TOOL_ROUNDS"]

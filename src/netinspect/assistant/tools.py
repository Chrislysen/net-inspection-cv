"""Read-only tools the inspection assistant may call.

Every tool reads a real artifact produced by this repo — a results JSON, a
telemetry parquet, the evidence ledger — and returns JSON. None of them
compute a new number on the fly, so an answer can always be traced back to a
file on disk. Each result carries an ``artifact`` field naming its source,
which is what makes the assistant's citations checkable rather than decorative.

Tools are decorated with ``@beta_tool``: the Anthropic SDK derives each JSON
schema from the signature and docstring, so the docstrings below are part of
the interface the model sees, not just developer documentation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..envelope import EnvelopeGate, EnvelopeSpec
from ..utils import get_logger, optional_import
from .evidence import CLAIMS_BY_TOPIC, LEDGER, load_result, unvalidated_topics

LOGGER = get_logger()

ENVELOPE_RESULT = "reports/results/operating_envelope/operating_envelope.json"
FRAME_CONDITIONS = "reports/results/operating_envelope/frame_conditions.parquet"
TELEMETRY_DIR = "data/processed/telemetry"


def _err(message: str, **extra: Any) -> str:
    """Uniform error payload — the model is told plainly that data is missing."""
    return json.dumps({"error": message, **extra})


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


# --------------------------------------------------------------------------- #
# Tool implementations (plain functions, wrapped below)
# --------------------------------------------------------------------------- #
def _list_inspections() -> str:
    data = load_result(ENVELOPE_RESULT)
    if data is None:
        return _err("No inspection analysis found. Run "
                    "scripts/analyze_operating_envelope.py first.",
                    expected_artifact=ENVELOPE_RESULT)
    profiles = data.get("flight_profiles", {})
    return _ok({
        "artifact": ENVELOPE_RESULT,
        "clips": [
            {"clip": clip, "day": clip[:10], "frames": p.get("frames"),
             "commanded_standoff_m": p.get("commanded_standoff_m"),
             "achieved_standoff_m": p.get("achieved_standoff_m"),
             "net_speed_ms": p.get("net_speed_ms"),
             "depth_m": p.get("depth_m"),
             "water_temperature_c": p.get("temperature_c"),
             "capture_sharpness": p.get("sharpness")}
            for clip, p in profiles.items()],
        "note": ("All clips show UNDAMAGED net from one SINTEF SOLAQUA site over "
                 "two days in August 2024."),
    })


def _get_false_alarm_analysis(model: str) -> str:
    data = load_result(ENVELOPE_RESULT)
    if data is None:
        return _err("No inspection analysis found.", expected_artifact=ENVELOPE_RESULT)
    models = data.get("models", {})
    if model not in models:
        return _err(f"Unknown model {model!r}.", available_models=list(models))
    m = models[model]
    return _ok({
        "artifact": ENVELOPE_RESULT,
        "model": model,
        "false_alarm_frame_rate": m["overall_naive"]["rate"],
        "naive_per_frame_ci95": m["overall_naive"]["ci95"],
        "clip_clustered_ci95": m["overall_clustered"].get("ci95_clustered"),
        "intracluster_correlation": m["clustering"].get("icc"),
        "effective_sample_size": m["clustering"].get("effective_n"),
        "naive_sample_size": m["clustering"].get("naive_n"),
        "by_clip": {c: {"rate": s["rate"], "frames": s["n"]}
                    for c, s in m["by_clip"].items()},
        "strongest_correlates": {
            k: {"r": v.get("r"), "p": v.get("p")}
            for k, v in sorted(m["condition_correlations"].items(),
                               key=lambda kv: -abs(kv[1].get("r") or 0))[:4]},
        "interpretation_required": (
            "This is false-alarm rate on real UNDAMAGED net. It says nothing about "
            "whether real damage would be detected. Quote the clip-clustered "
            "interval, not the naive per-frame one — frames within a clip are "
            "correlated and the effective sample size is far below the frame count."),
    })


def _get_evidence(topic: str) -> str:
    if topic in ("", "all", "list"):
        return _ok({"topics": [c.topic for c in LEDGER],
                    "unvalidated": unvalidated_topics()})
    claim = CLAIMS_BY_TOPIC.get(topic)
    if claim is None:
        return _err(f"Unknown topic {topic!r}.",
                    available_topics=[c.topic for c in LEDGER])
    return _ok(claim.to_dict())


def _query_telemetry(clip: str, stream: str, column: str, statistic: str) -> str:
    pd = optional_import("pandas")
    if pd is None:
        return _err("pandas is not installed.")
    path = Path(TELEMETRY_DIR) / f"{clip}__{stream}.parquet"
    if not path.exists():
        available = sorted(p.stem for p in Path(TELEMETRY_DIR).glob(f"{clip}__*.parquet")) \
            if Path(TELEMETRY_DIR).exists() else []
        return _err(f"No telemetry for clip {clip!r} stream {stream!r}. "
                    "Run scripts/extract_telemetry.py --all first.",
                    available=available)
    df = pd.read_parquet(path)
    if column not in df.columns:
        return _err(f"Unknown column {column!r}.", available_columns=list(df.columns))
    series = df[column].dropna()
    if series.empty:
        return _err(f"Column {column!r} has no values for this clip.")
    stats = {
        "mean": float(series.mean()), "min": float(series.min()),
        "max": float(series.max()), "std": float(series.std()),
        "median": float(series.median()), "count": int(series.size),
    }
    if statistic not in stats and statistic != "all":
        return _err(f"Unknown statistic {statistic!r}.",
                    available_statistics=sorted(stats))
    return _ok({
        "artifact": str(path),
        "clip": clip, "stream": stream, "column": column,
        "result": stats if statistic == "all" else {statistic: stats[statistic]},
        "source_topic": str(df["source_topic"].iloc[0]) if "source_topic" in df else None,
    })


def _check_capture_conditions(standoff_m: float, speed_ms: float,
                              net_lock: bool) -> str:
    # Bounds are the range actually flown across the SOLAQUA clips; the analysis
    # rejected standoff as a driver of false alarms, so this reports whether
    # conditions are *represented in the evaluated data*, not whether the model
    # is accurate there.
    spec = EnvelopeSpec(
        standoff_min_m=0.19, standoff_max_m=1.54, speed_max_ms=0.35,
        model="observed-data-range",
        evidence={"source": ENVELOPE_RESULT,
                  "meaning": "range of conditions present in the evaluated sample"})
    verdict = EnvelopeGate(spec).check(standoff_m=standoff_m, speed_ms=speed_ms,
                                       locked=net_lock)
    d = verdict.to_dict()
    d.update({
        "artifact": ENVELOPE_RESULT,
        "spec_meaning": (
            "'in_envelope' means these capture conditions fall inside the range "
            "actually observed in the evaluated data — NOT that detection accuracy "
            "is guaranteed there. Standoff was tested and rejected as a driver of "
            "false alarms; scene identity dominates."),
    })
    return _ok(d)


def _summarise_inspection_findings() -> str:
    data = load_result(ENVELOPE_RESULT)
    if data is None:
        return _err("No inspection analysis found.", expected_artifact=ENVELOPE_RESULT)
    return _ok({
        "artifact": ENVELOPE_RESULT,
        "frames_analysed": data["data"]["frames_analysed"],
        "clips": data["data"]["clips"],
        "conclusions": data.get("conclusions", []),
        "caveats": data.get("caveats", []),
        "ensemble_mechanism": data.get("ensemble_mechanism", {}).get("interpretation"),
    })


# --------------------------------------------------------------------------- #
# Provider-neutral tool specifications
#
# One declarative spec per tool, adapted per backend below. The schemas live
# here once rather than per provider: a tool whose Anthropic and Ollama
# descriptions drift apart is a bug that shows up as a behavioural difference
# between backends and gets misread as a model difference.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """One tool: its name, what the model is told, its schema, and its code."""
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def json_schema(self) -> dict[str, Any]:
        """The JSON Schema object both providers expect for parameters."""
        required = [k for k, v in self.parameters.items() if v.get("required", True)]
        return {
            "type": "object",
            "properties": {
                k: {kk: vv for kk, vv in v.items() if kk != "required"}
                for k, v in self.parameters.items()
            },
            "required": required,
            "additionalProperties": False,
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_inspections",
        "List every inspection clip with its flight profile: the ROV's commanded "
        "and achieved standoff from the net, sweep speed, depth, water temperature "
        "and mean capture sharpness. Call this first to find out which clips exist.",
        {}, lambda: _list_inspections()),
    ToolSpec(
        "get_false_alarm_analysis",
        "Get measured false-alarm behaviour for one detection model on real "
        "undamaged net. Returns the false-alarm frame rate, both naive per-frame "
        "and clip-clustered confidence intervals, the intra-cluster correlation "
        "and effective sample size, a per-clip breakdown, and the strongest "
        "capture-condition correlates.",
        {"model": {"type": "string", "description": "One of det_v1, seg_v3, seg_gpu."}},
        lambda model: _get_false_alarm_analysis(model)),
    ToolSpec(
        "get_evidence",
        "Look up what this project may and may not claim about a topic. Returns "
        "the evidence level (measured on real data, measured on a synthetic proxy, "
        "inferred, or NOT VALIDATED), the backing artifact and any caveat. Pass "
        "'all' to list every topic. Always call this before claiming a capability.",
        {"topic": {"type": "string", "description": "A ledger topic, or 'all'."}},
        lambda topic: _get_evidence(topic)),
    ToolSpec(
        "query_telemetry",
        "Query real ROV sensor telemetry for one inspection clip. Streams: "
        "net_plane (net_distance, net_heading, net_pitch, net_lock, net_vel_u/v/w), "
        "depth_temp (depth, temperature, pressure), dvl (vel_x/y/z, altitude), "
        "attitude (roll, pitch, yaw), battery, setpoint (commanded values).",
        {"clip": {"type": "string", "description": "Clip id, e.g. 2024-08-22_14-47-39."},
         "stream": {"type": "string", "description": "Telemetry stream name."},
         "column": {"type": "string", "description": "Column within that stream."},
         "statistic": {"type": "string", "required": False,
                       "description": "mean, min, max, std, median, count, or all."}},
        lambda clip, stream, column, statistic="all":
            _query_telemetry(clip, stream, column, statistic)),
    ToolSpec(
        "check_capture_conditions",
        "Check whether a given standoff distance and sweep speed fall inside the "
        "range of conditions the models were characterised on. Being inside the "
        "range does NOT imply detection accuracy there.",
        {"standoff_m": {"type": "number", "description": "Distance from the net in metres."},
         "speed_ms": {"type": "number", "description": "Net-relative sweep speed in m/s."},
         "net_lock": {"type": "boolean", "required": False,
                      "description": "Whether the net-plane estimator reported a lock."}},
        lambda standoff_m, speed_ms, net_lock=True:
            _check_capture_conditions(standoff_m, speed_ms, net_lock)),
    ToolSpec(
        "summarise_inspection_findings",
        "Get the headline conclusions and caveats from the inspection analysis: "
        "what it established, what it rejected, and the limitations on every number.",
        {}, lambda: _summarise_inspection_findings()),
)

SPECS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}
TOOL_NAMES = tuple(t.name for t in TOOL_SPECS)


def anthropic_tools() -> list[dict[str, Any]]:
    """Tool definitions in Anthropic Messages API shape."""
    return [{"name": t.name, "description": t.description,
             "input_schema": t.json_schema()} for t in TOOL_SPECS]


def ollama_tools() -> list[dict[str, Any]]:
    """Tool definitions in the OpenAI-style shape Ollama expects."""
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.json_schema()}} for t in TOOL_SPECS]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute one tool by name, returning its JSON string result.

    Unknown tools and bad arguments come back as an error payload rather than an
    exception: a model that invents a tool name should be told so and allowed to
    recover, not crash the run.
    """
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        return _err(f"Unknown tool {name!r}.", available_tools=list(TOOL_NAMES))
    try:
        return spec.fn(**(arguments or {}))
    except TypeError as exc:
        return _err(f"Bad arguments for {name}: {exc}",
                    expected_schema=spec.json_schema())
    except Exception as exc:  # a tool fault must not kill the conversation
        LOGGER.warning("Tool %s raised: %s", name, exc)
        return _err(f"Tool {name} failed: {exc}")


__all__ = ["ToolSpec", "TOOL_SPECS", "SPECS_BY_NAME", "TOOL_NAMES",
           "anthropic_tools", "ollama_tools", "call_tool",
           "ENVELOPE_RESULT", "TELEMETRY_DIR"]

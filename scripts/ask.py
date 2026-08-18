"""Ask the inspection assistant a question about this project's real data.

The assistant answers from artifacts in this repo — inspection results, ROV
telemetry, the evidence ledger — and cites the file behind each number. When a
question reaches for something the project has never validated (most often:
performance on real damage), it says so instead of answering with a
synthetic-proxy figure. See ``src/netinspect/assistant/agent.py``.

Needs ``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile) and the assistant
extra: ``pip install -e '.[assistant]'``.

Examples
--------
    python scripts/ask.py "Which clip produced the most false alarms?"
    python scripts/ask.py "How accurate is this on real damage?"
    python scripts/ask.py --interactive
    python scripts/ask.py --show-ledger
"""
from __future__ import annotations

import argparse
import sys

import _common  # noqa: F401

from netinspect.assistant.evidence import render_for_prompt


def _print_answer(answer) -> None:
    print()
    print(answer.text)
    print()
    if answer.tools_used:
        print(f"  tools:     {', '.join(dict.fromkeys(answer.tools_used))}")
    if answer.artifacts_cited:
        print(f"  artifacts: {', '.join(answer.artifacts_cited)}")
    g = answer.grounding
    if g.get("question_touches_unvalidated_capability"):
        state = "stated" if g["answer_states_boundary"] else "MISSING"
        print(f"  boundary:  {state} (question reached for unvalidated capability)")
    if not answer.tools_used:
        print("  warning:   answered without calling a tool")
    if answer.usage.get("output_tokens"):
        print(f"  tokens:    {answer.usage['input_tokens']} in / "
              f"{answer.usage['output_tokens']} out")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="The question to ask")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Keep asking in a loop")
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "ollama"])
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--show-ledger", action="store_true",
                    help="Print the evidence ledger the assistant is grounded in, then exit")
    args = ap.parse_args()

    if args.show_ledger:
        print(render_for_prompt())
        return

    if not args.question and not args.interactive:
        ap.error("Provide a question, or use --interactive.")

    from netinspect.assistant import InspectionAssistant

    kwargs = {"backend": args.backend}
    if args.backend == "anthropic":
        kwargs["effort"] = args.effort
    if args.model:
        kwargs["model"] = args.model
    try:
        assistant = InspectionAssistant(**kwargs)
    except RuntimeError as exc:
        sys.exit(str(exc))

    if args.question:
        _print_answer(assistant.ask(" ".join(args.question)))
        if not args.interactive:
            return

    print("Inspection assistant. Ctrl-C or an empty line to exit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        _print_answer(assistant.ask(q))


if __name__ == "__main__":
    main()

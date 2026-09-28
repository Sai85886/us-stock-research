"""Ask the LangGraph supervisor to route a research question."""

from __future__ import annotations

import argparse
import json
import sys

from stock_research.agents.supervisor import run_research
from stock_research.config import get_settings
from stock_research.observability import configure_tracing, tracing_status


def _run_with_optional_tracing(tracing_on: bool, question: str, settings, max_handoffs: int):
    if tracing_on:
        return run_research(question, settings, max_handoffs=max_handoffs)
    try:
        from langsmith.run_helpers import tracing_context
    except ImportError:
        return run_research(question, settings, max_handoffs=max_handoffs)
    with tracing_context(enabled=False):
        return run_research(question, settings, max_handoffs=max_handoffs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route a question through the LangGraph research supervisor."
    )
    parser.add_argument("question", nargs="?", default="", help="Research question")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show routing decision and tool trace",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON",
    )
    parser.add_argument(
        "--max-handoffs",
        type=int,
        default=3,
        help="Maximum specialist handoffs (default: 3)",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--trace",
        action="store_true",
        help="Force-enable LangSmith tracing for this run",
    )
    trace_group.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable LangSmith tracing for this run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    question = args.question.strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("A question is required.", file=sys.stderr)
        return 1

    enabled = True if args.trace else False if args.no_trace else None
    try:
        tracing_on = configure_tracing(settings, enabled=enabled)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.verbose and tracing_on:
        status = tracing_status()
        print(f"LangSmith tracing: on (project={status['project']})")

    try:
        result = _run_with_optional_tracing(
            tracing_on,
            question,
            settings,
            args.max_handoffs,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show clean errors
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "question": result.question,
            "routes": result.routes,
            "route": result.route,
            "route_reason": result.route_reason,
            "answer": result.answer,
            "specialist_answers": result.specialist_answers,
            "agent_steps": [
                {
                    "thought": step.thought,
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "observation": step.observation,
                }
                for step in result.agent_steps
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.verbose:
        routes = " -> ".join(result.routes) if result.routes else "(none)"
        print(f"Routes: {routes}")
        print(f"Last decision: {result.route_reason}")
        if result.specialist_answers:
            print("Specialists:")
            for item in result.specialist_answers:
                preview = str(item.get("answer") or "")[:180].replace("\n", " ")
                print(f"- {item.get('agent')}: {preview}...")
        if result.agent_steps:
            print("Tool trace:")
            for index, step in enumerate(result.agent_steps, start=1):
                print(f"{index}. {step.tool_name}({json.dumps(step.tool_args)})")
                observation = step.observation or ""
                preview = observation[:300].replace("\n", " ")
                print(f"   -> {preview}{'...' if len(observation) > 300 else ''}")
        print()

    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

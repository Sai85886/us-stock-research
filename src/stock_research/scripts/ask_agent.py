"""Run a specialist ReAct agent (rag, market, or web)."""

from __future__ import annotations

import argparse
import json
import sys

from stock_research.agents.specialists import SPECIALIST_BUILDERS, build_specialist
from stock_research.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a specialist ReAct agent.")
    parser.add_argument(
        "agent",
        choices=sorted(SPECIALIST_BUILDERS),
        help="Which specialist to run",
    )
    parser.add_argument("question", nargs="?", default="", help="Question to ask")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Max tool-calling rounds (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool calls and observations",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full agent result as JSON",
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

    try:
        agent = build_specialist(args.agent, settings)
        agent.max_steps = args.max_steps
        result = agent.run(question)
    except Exception as exc:  # noqa: BLE001 - CLI should show clean errors
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "agent": result.agent,
            "question": result.question,
            "answer": result.answer,
            "steps": [
                {
                    "thought": step.thought,
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "observation": step.observation,
                }
                for step in result.steps
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.verbose and result.steps:
        print("Tool trace:")
        for index, step in enumerate(result.steps, start=1):
            print(f"{index}. {step.tool_name}({json.dumps(step.tool_args)})")
            observation = step.observation or ""
            preview = observation[:300].replace("\n", " ")
            print(f"   -> {preview}{'...' if len(observation) > 300 else ''}")
        print()

    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

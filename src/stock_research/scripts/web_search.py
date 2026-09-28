"""CLI for Tavily web and news search."""

from __future__ import annotations

import argparse
import json
import sys

from stock_research.config import get_settings
from stock_research.tools.web import search_news, search_web


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the web/news with Tavily.")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument(
        "--news",
        action="store_true",
        help="Prefer recent news results",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback window in days for --news (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )
    return parser.parse_args()


def print_results(payload: dict) -> None:
    print(f"Query: {payload['query']}")
    print(f"Topic: {payload['topic']}")
    print(f"As of: {payload['as_of']}\n")

    results = payload.get("results") or []
    if not results:
        print("No results.")
        return

    for index, item in enumerate(results, start=1):
        score = item.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        published = item.get("published_date") or "n/a"
        content = str(item.get("content") or "").replace("\n", " ")
        preview = content[:220] + ("..." if len(content) > 220 else "")
        print(
            f"{index}. {item.get('title') or '(untitled)'}\n"
            f"   {item.get('url')}\n"
            f"   score={score_text} | published={published}\n"
            f"   {preview}\n"
        )


def main() -> int:
    args = parse_args()
    settings = get_settings()

    query = args.query.strip()
    if not query:
        query = input("Query: ").strip()
    if not query:
        print("A query is required.", file=sys.stderr)
        return 1

    try:
        if args.news:
            payload = search_news(
                query,
                api_key=settings.tavily_api_key,
                max_results=args.max_results,
                days=args.days,
            )
        else:
            payload = search_web(
                query,
                api_key=settings.tavily_api_key,
                max_results=args.max_results,
            )
    except Exception as exc:  # noqa: BLE001 - CLI should show clean errors
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_results(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

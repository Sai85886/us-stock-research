"""Unified CLI entrypoint for us-stock-research."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from stock_research import __version__
from stock_research.scripts import (
    ask_agent,
    ask_filings,
    chunk_filings,
    download_filings,
    embed_filings,
    market_quote,
    research,
    web_search,
)

CommandHandler = Callable[[], int]


def status_main() -> int:
    """Print a quick health check for local setup."""
    from stock_research.config import get_settings

    settings = get_settings()

    def _flag(value: str) -> str:
        return "set" if value.strip() else "missing"

    print(f"us-stock-research {__version__}")
    print("\nConfig:")
    print(f"  GROQ_API_KEY: {_flag(settings.groq_api_key)}")
    print(f"  TAVILY_API_KEY: {_flag(settings.tavily_api_key)}")
    print(f"  LANGCHAIN_API_KEY: {_flag(settings.langchain_api_key)}")
    print(f"  SEC_EDGAR_USER_AGENT: {_flag(settings.sec_edgar_user_agent)}")
    print(f"  GROQ_MODEL: {settings.groq_model}")
    print(f"  EMBEDDING_MODEL: {settings.embedding_model}")
    print(f"  tracing default: {settings.langchain_tracing_v2}")

    raw_dir = Path(settings.raw_data_dir)
    processed_dir = Path(settings.processed_data_dir)
    chroma_dir = Path(settings.chroma_dir)

    print("\nData:")
    print(f"  raw: {raw_dir} ({'yes' if raw_dir.exists() else 'no'})")
    print(f"  processed: {processed_dir} ({'yes' if processed_dir.exists() else 'no'})")
    print(f"  chroma: {chroma_dir} ({'yes' if chroma_dir.exists() else 'no'})")

    if chroma_dir.exists():
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_or_create_collection(name=settings.chroma_collection)
            print(f"  chroma vectors: {collection.count()}")
        except Exception as exc:  # noqa: BLE001 - status should keep going
            print(f"  chroma vectors: unavailable ({exc})")

    print("\nUseful commands:")
    print('  stock-research research "What is AAPL price?" --verbose')
    print("  stock-research download --tickers AAPL MSFT JPM")
    print("  stock-research status")
    return 0


COMMANDS: dict[str, tuple[str, CommandHandler]] = {
    "download": ("Download latest SEC 10-K filings", download_filings.main),
    "chunk": ("Chunk downloaded filings into JSONL", chunk_filings.main),
    "embed": ("Embed chunks into Chroma", embed_filings.main),
    "ask-filings": ("Ask a basic RAG question over filings", ask_filings.main),
    "quote": ("Fetch live market quote / fundamentals", market_quote.main),
    "web": ("Search the web/news with Tavily", web_search.main),
    "agent": ("Run one specialist ReAct agent", ask_agent.main),
    "research": ("Ask the multi-agent supervisor", research.main),
    "status": ("Show local setup / data status", status_main),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-research",
        description="Unified CLI for the multi-agent US stock research system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    for name, (help_text, _) in COMMANDS.items():
        sub = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
            add_help=False,
        )
        sub.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the subcommand",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help"}:
        _build_parser().print_help()
        print("\nExamples:")
        print('  stock-research research "What are Apple risk factors?" --verbose')
        print("  stock-research quote AAPL --fundamentals")
        print("  stock-research download && stock-research chunk && stock-research embed")
        return 0

    if argv[0] in {"-V", "--version"}:
        print(f"stock-research {__version__}")
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(f"Available: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2

    forwarded = argv[1:]
    # Drop a leading "--" separator sometimes used before subcommand args.
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    _, handler = COMMANDS[command]
    original_argv = sys.argv
    try:
        sys.argv = [f"stock-research-{command}", *forwarded]
        result = handler()
        return int(result or 0)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

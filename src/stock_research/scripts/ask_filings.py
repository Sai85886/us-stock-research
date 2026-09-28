"""Ask questions over embedded SEC 10-K filings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_research.config import get_settings
from stock_research.ingestion.embedder import TextEmbedder
from stock_research.rag.chain import FilingRAG, GroqChatCompleter
from stock_research.vectorstore.chroma_store import ChromaStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over embedded SEC filings.")
    parser.add_argument("question", nargs="?", default="", help="Question to ask")
    parser.add_argument(
        "--ticker",
        default=None,
        help="Optional ticker filter (e.g. AAPL)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help="Chroma persistence directory (default: data/chroma)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default from config)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Groq model name (default from config)",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved source excerpts after the answer",
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

    chroma_dir = args.chroma_dir or Path(settings.chroma_dir)
    collection = args.collection or settings.chroma_collection
    model = args.model or settings.groq_model

    if not chroma_dir.exists():
        print(
            f"Chroma directory not found: {chroma_dir}\n"
            "Run embed-filings first.",
            file=sys.stderr,
        )
        return 1

    try:
        llm = GroqChatCompleter(api_key=settings.groq_api_key, model=model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    store = ChromaStore(
        persist_dir=chroma_dir,
        embedder=TextEmbedder(settings.embedding_model),
        collection_name=collection,
    )
    if store.count == 0:
        print(
            f"Chroma collection '{collection}' is empty. Run embed-filings first.",
            file=sys.stderr,
        )
        return 1

    rag = FilingRAG(store=store, llm=llm)
    result = rag.ask(question, n_results=args.n_results, ticker=args.ticker)

    print(result.answer)

    if args.show_sources:
        print("\nSources:")
        if not result.sources:
            print("  (none)")
        for index, hit in enumerate(result.sources, start=1):
            metadata = hit.get("metadata") or {}
            preview = str(hit.get("text") or "")[:180].replace("\n", " ")
            distance = hit.get("distance")
            distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
            print(
                f"{index}. {metadata.get('ticker')} | {metadata.get('section')} | "
                f"filed {metadata.get('filing_date')} | distance={distance_text}\n"
                f"   {preview}..."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

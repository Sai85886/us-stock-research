"""Embed chunked SEC filings into a local ChromaDB store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_research.config import get_settings
from stock_research.ingestion.embedder import DEFAULT_EMBEDDING_MODEL, TextEmbedder
from stock_research.ingestion.models import DocumentChunk
from stock_research.vectorstore.chroma_store import ChromaStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunked SEC filings into ChromaDB.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing chunks.jsonl files (default: data/processed)",
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
        help=f"SentenceTransformer model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before embedding",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Optional smoke-test query after embedding",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Number of results for --query",
    )
    return parser.parse_args()


def discover_chunk_files(processed_dir: Path) -> list[Path]:
    """Find chunks.jsonl files under processed_dir."""
    return sorted(processed_dir.glob("*/chunks.jsonl"))


def load_chunks_jsonl(path: Path) -> list[DocumentChunk]:
    """Load DocumentChunk rows from a JSONL file."""
    chunks: list[DocumentChunk] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                chunks.append(
                    DocumentChunk(
                        chunk_id=payload["chunk_id"],
                        chunk_index=int(payload["chunk_index"]),
                        text=payload["text"],
                        ticker=payload["ticker"],
                        company=payload["company"],
                        cik=payload["cik"],
                        form=payload["form"],
                        filing_date=payload["filing_date"],
                        section=payload["section"],
                        source_file=payload["source_file"],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing field {exc} in {path} line {line_number}") from exc
    return chunks


def main() -> int:
    args = parse_args()
    settings = get_settings()

    processed_dir = args.processed_dir or Path(settings.processed_data_dir)
    chroma_dir = args.chroma_dir or Path(settings.chroma_dir)
    collection = args.collection or settings.chroma_collection
    model_name = args.model or settings.embedding_model

    if not processed_dir.exists():
        print(f"Processed data directory not found: {processed_dir}", file=sys.stderr)
        return 1

    chunk_files = discover_chunk_files(processed_dir)
    if not chunk_files:
        print(f"No chunks.jsonl files found in {processed_dir}", file=sys.stderr)
        return 1

    print(f"Reading chunks from: {processed_dir.resolve()}")
    print(f"Writing vectors to: {chroma_dir.resolve()}")
    print(f"Model: {model_name}")
    print(f"Collection: {collection}\n")

    embedder = TextEmbedder(model_name)
    store = ChromaStore(
        persist_dir=chroma_dir,
        embedder=embedder,
        collection_name=collection,
    )

    if args.reset:
        store.reset()
        print("Reset existing collection.\n")

    total_chunks = 0
    for chunk_path in chunk_files:
        chunks = load_chunks_jsonl(chunk_path)
        written = store.upsert_chunks(chunks, batch_size=args.batch_size)
        total_chunks += written
        ticker = chunks[0].ticker if chunks else chunk_path.parent.name
        print(f"[OK] {ticker} | upserted {written} chunks from {chunk_path.name}")

    print(f"\nCollection now has {store.count} vectors ({total_chunks} upserted this run).")

    if args.query.strip():
        print(f"\nQuery: {args.query!r}")
        hits = store.query(args.query, n_results=args.n_results)
        if not hits:
            print("No results.")
        for rank, hit in enumerate(hits, start=1):
            metadata = hit["metadata"]
            preview = hit["text"][:180].replace("\n", " ")
            print(
                f"{rank}. {metadata.get('ticker')} | {metadata.get('section')} | "
                f"distance={hit['distance']:.4f}\n   {preview}..."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

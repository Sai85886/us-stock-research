"""Chunk downloaded SEC 10-K filings for RAG ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from stock_research.config import get_settings
from stock_research.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_filing,
)
from stock_research.ingestion.loader import discover_filings, load_filing, load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk downloaded SEC 10-K filings.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing downloaded filings (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for chunked output (default: data/processed)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N chunks from each filing after processing",
    )
    return parser.parse_args()


def write_chunks(output_path: Path, chunks: list) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict()) + "\n")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    raw_dir = args.raw_dir or Path(settings.raw_data_dir)
    output_dir = args.output_dir or Path(settings.processed_data_dir)

    if not raw_dir.exists():
        print(f"Raw data directory not found: {raw_dir}", file=sys.stderr)
        return 1

    filings = discover_filings(raw_dir)
    if not filings:
        print(f"No filings found in {raw_dir}", file=sys.stderr)
        return 1

    print(f"Reading filings from: {raw_dir.resolve()}")
    print(f"Writing chunks to: {output_dir.resolve()}\n")

    total_chunks = 0

    for metadata_path, filing_path in filings:
        metadata = load_metadata(metadata_path)
        loaded = load_filing(filing_path, metadata)
        chunks = chunk_filing(
            loaded,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        ticker_dir = output_dir / metadata["ticker"]
        chunks_path = ticker_dir / "chunks.jsonl"
        summary_path = ticker_dir / "summary.json"

        write_chunks(chunks_path, chunks)

        section_counts = Counter(chunk.section for chunk in chunks)
        summary = {
            "ticker": metadata["ticker"],
            "company": metadata["company"],
            "filing_date": metadata["filing_date"],
            "source_file": str(filing_path),
            "chunk_count": len(chunks),
            "section_counts": dict(section_counts),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        total_chunks += len(chunks)
        print(
            f"[OK] {metadata['ticker']} | {len(chunks)} chunks | "
            f"{len(loaded.sections)} sections"
        )
        for section_name, count in sorted(section_counts.items()):
            print(f"     - {section_name}: {count}")

        if args.preview > 0:
            print("\nPreview:")
            for chunk in chunks[: args.preview]:
                preview_text = chunk.text[:200].replace("\n", " ")
                print(
                    f"  [{chunk.chunk_index}] {chunk.section} | "
                    f"{len(chunk.text)} chars | {preview_text}..."
                )
            print()

    print(f"\nCreated {total_chunks} chunks across {len(filings)} filings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

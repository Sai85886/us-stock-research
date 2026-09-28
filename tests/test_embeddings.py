"""Tests for embedding helpers and Chroma store."""

from __future__ import annotations

import json
from pathlib import Path

from stock_research.ingestion.models import DocumentChunk
from stock_research.scripts.embed_filings import discover_chunk_files, load_chunks_jsonl
from stock_research.vectorstore.chroma_store import ChromaStore


class FakeEmbedder:
    """Deterministic tiny embedder for offline tests."""

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            # Cheap bag-of-signal vector so similar strings land closer.
            lowered = text.lower()
            vectors.append(
                [
                    float(len(text) % 97) / 97.0,
                    1.0 if "risk" in lowered else 0.0,
                    1.0 if "revenue" in lowered else 0.0,
                    1.0 if "apple" in lowered else 0.0,
                ]
            )
        return vectors

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]


def _chunk(chunk_id: str, text: str, section: str = "Item 1A - Risk Factors") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=int(chunk_id.rsplit("_", 1)[-1]),
        text=text,
        ticker="AAPL",
        company="Apple Inc.",
        cik="0000320193",
        form="10-K",
        filing_date="2025-10-31",
        section=section,
        source_file="data/raw/AAPL/10-K_2025-10-31.htm",
    )


def test_load_chunks_jsonl_and_discover(tmp_path: Path):
    ticker_dir = tmp_path / "AAPL"
    ticker_dir.mkdir()
    chunks_path = ticker_dir / "chunks.jsonl"

    rows = [
        _chunk("AAPL_2025-10-31_0000", "Risk factors include competition."),
        _chunk("AAPL_2025-10-31_0001", "Revenue increased year over year.", "Item 7 - MD&A"),
    ]
    with chunks_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict()) + "\n")

    discovered = discover_chunk_files(tmp_path)
    assert discovered == [chunks_path]

    loaded = load_chunks_jsonl(chunks_path)
    assert len(loaded) == 2
    assert loaded[0].ticker == "AAPL"
    assert loaded[1].section == "Item 7 - MD&A"


def test_chroma_store_upsert_and_query(tmp_path: Path):
    store = ChromaStore(
        persist_dir=tmp_path / "chroma",
        embedder=FakeEmbedder(),
        collection_name="test_filings",
    )

    chunks = [
        _chunk("AAPL_2025-10-31_0000", "Apple faces significant risk from competitors."),
        _chunk(
            "AAPL_2025-10-31_0001",
            "Apple reported higher revenue this fiscal year.",
            "Item 7 - MD&A",
        ),
    ]
    written = store.upsert_chunks(chunks)
    assert written == 2
    assert store.count == 2

    hits = store.query("what are the risk factors?", n_results=1)
    assert len(hits) == 1
    assert "risk" in hits[0]["text"].lower()
    assert hits[0]["metadata"]["ticker"] == "AAPL"

"""ChromaDB persistence for embedded filing chunks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from stock_research.ingestion.models import DocumentChunk

DEFAULT_COLLECTION = "sec_10k_filings"


class Embedder(Protocol):
    """Minimal embedding interface used by ChromaStore."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def chunk_to_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    """Convert chunk fields into Chroma-safe metadata."""
    return {
        "chunk_index": chunk.chunk_index,
        "ticker": chunk.ticker,
        "company": chunk.company,
        "cik": chunk.cik,
        "form": chunk.form,
        "filing_date": chunk.filing_date,
        "section": chunk.section,
        "source_file": chunk.source_file,
    }


class ChromaStore:
    """Persistent Chroma collection backed by a local embedder."""

    def __init__(
        self,
        persist_dir: str | Path,
        embedder: Embedder,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.collection_name = collection_name

        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: Sequence[DocumentChunk], batch_size: int = 64) -> int:
        """Embed and upsert chunks. Returns number of chunks written."""
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedder.embed_documents(texts)
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[chunk_to_metadata(chunk) for chunk in batch],
            )
            total += len(batch)
        return total

    def query(
        self,
        text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over stored chunks."""
        if self.count == 0:
            return []

        n_results = min(n_results, self.count)
        query_embedding = self.embedder.embed_query(text)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)
        hits: list[dict[str, Any]] = []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for index, chunk_id in enumerate(ids):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index],
                    "metadata": metadatas[index] or {},
                    "distance": distances[index],
                }
            )
        return hits

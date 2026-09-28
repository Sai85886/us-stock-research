"""Tests for the basic filing RAG chain."""

from __future__ import annotations

from stock_research.rag.chain import FilingRAG, format_context


class FakeStore:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.last_query = None

    def query(self, text: str, n_results: int = 5, where=None):
        self.last_query = {"text": text, "n_results": n_results, "where": where}
        return self.hits[:n_results]


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "Apple cites competition and supply-chain constraints as key risks."


def test_format_context_includes_metadata():
    context = format_context(
        [
            {
                "chunk_id": "AAPL_1",
                "text": "Competition is intense in smartphone markets.",
                "metadata": {
                    "ticker": "AAPL",
                    "company": "Apple Inc.",
                    "section": "Item 1A - Risk Factors",
                    "filing_date": "2025-10-31",
                },
                "distance": 0.21,
            }
        ]
    )
    assert "AAPL" in context
    assert "Item 1A - Risk Factors" in context
    assert "Competition is intense" in context


def test_filing_rag_ask_uses_retrieved_context_and_ticker_filter():
    hits = [
        {
            "chunk_id": "AAPL_1",
            "text": "Risk factors include competition and component shortages.",
            "metadata": {
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "section": "Item 1A - Risk Factors",
                "filing_date": "2025-10-31",
            },
            "distance": 0.3,
        }
    ]
    store = FakeStore(hits)
    llm = FakeLLM()
    rag = FilingRAG(store=store, llm=llm)

    result = rag.ask("What are Apple's risks?", n_results=3, ticker="aapl")

    assert store.last_query == {
        "text": "What are Apple's risks?",
        "n_results": 3,
        "where": {"ticker": "AAPL"},
    }
    assert "competition" in result.answer.lower()
    assert result.sources == hits
    assert "CONTEXT:" in llm.calls[0][1]
    assert "Risk factors include competition" in llm.calls[0][1]

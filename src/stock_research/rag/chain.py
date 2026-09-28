"""Basic retrieve-then-generate RAG chain over Chroma-stored filings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from stock_research.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from stock_research.vectorstore.chroma_store import ChromaStore

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


@dataclass
class RAGAnswer:
    """Answer plus the retrieved evidence used to produce it."""

    question: str
    answer: str
    sources: list[dict[str, Any]]


class ChatCompleter(Protocol):
    """Minimal chat interface so tests can fake the LLM."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class GroqChatCompleter:
    """Thin wrapper around the Groq chat completions API."""

    def __init__(self, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> None:
        if not api_key.strip():
            raise ValueError(
                "GROQ_API_KEY is required. Set it in .env to use the RAG chain."
            )
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key.strip())

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


def format_context(hits: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into a prompt-friendly evidence block."""
    if not hits:
        return "No relevant excerpts were retrieved."

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata") or {}
        ticker = metadata.get("ticker", "UNKNOWN")
        section = metadata.get("section", "Unknown section")
        filing_date = metadata.get("filing_date", "unknown date")
        company = metadata.get("company", "")
        distance = hit.get("distance")
        distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
        text = str(hit.get("text") or "").strip()
        header = (
            f"[{index}] {ticker} | {company} | {section} | "
            f"filed {filing_date} | distance={distance_text}"
        )
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


class FilingRAG:
    """Retrieve filing chunks from Chroma and answer with an LLM."""

    def __init__(self, store: ChromaStore, llm: ChatCompleter) -> None:
        self.store = store
        self.llm = llm

    def ask(
        self,
        question: str,
        n_results: int = 5,
        ticker: str | None = None,
    ) -> RAGAnswer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")

        where = {"ticker": ticker.upper()} if ticker else None
        sources = self.store.query(cleaned, n_results=n_results, where=where)
        context = format_context(sources)
        user_prompt = build_user_prompt(cleaned, context)
        answer = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        return RAGAnswer(question=cleaned, answer=answer, sources=sources)

"""Specialist ReAct agents for filings, market data, and web/news."""

from __future__ import annotations

from pathlib import Path

from stock_research.agents.react import GroqToolChatModel, ReActAgent
from stock_research.agents.tools import build_market_tools, build_rag_tools, build_web_tools
from stock_research.config import Settings
from stock_research.ingestion.embedder import TextEmbedder
from stock_research.vectorstore.chroma_store import ChromaStore

RAG_SYSTEM_PROMPT = """You are the filings specialist for a US stock research system.
Use the search_filings tool to gather evidence from SEC 10-K excerpts before answering.
Answer only from tool results. If evidence is weak, say what is missing.
Cite ticker, section, and filing date when possible.
Do not give investment advice."""

MARKET_SYSTEM_PROMPT = """You are the market-data specialist for a US stock research system.
Use get_quote, get_fundamentals, and get_history to answer questions about prices and fundamentals.
Call tools before answering. Be concise and factual.
Do not give investment advice."""

WEB_SYSTEM_PROMPT = """You are the web/news specialist for a US stock research system.
Use search_news for recent headlines and search_web for broader context.
Summarize only what the tools return and mention source titles/URLs when useful.
Do not give investment advice."""


def build_rag_agent(settings: Settings, llm: GroqToolChatModel | None = None) -> ReActAgent:
    store = ChromaStore(
        persist_dir=Path(settings.chroma_dir),
        embedder=TextEmbedder(settings.embedding_model),
        collection_name=settings.chroma_collection,
    )
    if store.count == 0:
        raise ValueError(
            f"Chroma collection '{settings.chroma_collection}' is empty. Run embed-filings first."
        )
    model = llm or GroqToolChatModel(settings.groq_api_key, settings.groq_model)
    return ReActAgent(
        name="rag",
        system_prompt=RAG_SYSTEM_PROMPT,
        tools=build_rag_tools(store),
        llm=model,
    )


def build_market_agent(settings: Settings, llm: GroqToolChatModel | None = None) -> ReActAgent:
    model = llm or GroqToolChatModel(settings.groq_api_key, settings.groq_model)
    return ReActAgent(
        name="market",
        system_prompt=MARKET_SYSTEM_PROMPT,
        tools=build_market_tools(),
        llm=model,
    )


def build_web_agent(settings: Settings, llm: GroqToolChatModel | None = None) -> ReActAgent:
    model = llm or GroqToolChatModel(settings.groq_api_key, settings.groq_model)
    return ReActAgent(
        name="web",
        system_prompt=WEB_SYSTEM_PROMPT,
        tools=build_web_tools(settings.tavily_api_key),
        llm=model,
    )


SPECIALIST_BUILDERS = {
    "rag": build_rag_agent,
    "market": build_market_agent,
    "web": build_web_agent,
}


def build_specialist(name: str, settings: Settings) -> ReActAgent:
    key = name.strip().lower()
    builder = SPECIALIST_BUILDERS.get(key)
    if builder is None:
        supported = ", ".join(sorted(SPECIALIST_BUILDERS))
        raise ValueError(f"Unknown agent '{name}'. Supported: {supported}")
    return builder(settings)

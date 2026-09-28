"""Agent toolkit: tool definitions and executors for specialist ReAct agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from stock_research.rag.chain import format_context
from stock_research.tools.market import get_fundamentals, get_history, get_quote
from stock_research.tools.web import search_news, search_web
from stock_research.vectorstore.chroma_store import ChromaStore

ToolHandler = Callable[..., Any]


@dataclass
class AgentTool:
    """One callable tool exposed to a ReAct agent."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, raw_arguments: str | dict[str, Any]) -> str:
        if isinstance(raw_arguments, str):
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        else:
            args = raw_arguments
        if not isinstance(args, dict):
            raise TypeError("Tool arguments must be a JSON object")
        result = self.handler(**args)
        return json.dumps(result, default=str)


def build_rag_tools(store: ChromaStore) -> list[AgentTool]:
    def search_filings(
        query: str,
        ticker: str | None = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        where = {"ticker": ticker.upper()} if ticker else None
        hits = store.query(query, n_results=n_results, where=where)
        return {
            "query": query,
            "ticker": ticker.upper() if ticker else None,
            "result_count": len(hits),
            "context": format_context(hits),
            "hits": [
                {
                    "chunk_id": hit.get("chunk_id"),
                    "distance": hit.get("distance"),
                    "metadata": hit.get("metadata") or {},
                    "text": str(hit.get("text") or "")[:500],
                }
                for hit in hits
            ],
        }

    return [
        AgentTool(
            name="search_filings",
            description=(
                "Semantic search over embedded SEC 10-K filing chunks. "
                "Use for business description, risk factors, MD&A, and other filing content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional ticker filter such as AAPL",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5)",
                    },
                },
                "required": ["query"],
            },
            handler=search_filings,
        )
    ]


def build_market_tools() -> list[AgentTool]:
    return [
        AgentTool(
            name="get_quote",
            description="Get the latest price quote for a US stock ticker.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. AAPL",
                    }
                },
                "required": ["ticker"],
            },
            handler=lambda ticker: get_quote(ticker),
        ),
        AgentTool(
            name="get_fundamentals",
            description=(
                "Get company profile and key fundamentals for a ticker "
                "(sector, market cap, P/E, EPS, 52-week range, etc.)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. MSFT",
                    }
                },
                "required": ["ticker"],
            },
            handler=lambda ticker: get_fundamentals(ticker),
        ),
        AgentTool(
            name="get_history",
            description="Get recent OHLCV price history for a ticker.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol"},
                    "period": {
                        "type": "string",
                        "description": "History window such as 5d, 1mo, 3mo (default 1mo)",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Bar interval such as 1d (default 1d)",
                    },
                },
                "required": ["ticker"],
            },
            handler=lambda ticker, period="1mo", interval="1d": get_history(
                ticker, period=period, interval=interval
            ),
        ),
    ]


def build_web_tools(api_key: str) -> list[AgentTool]:
    def _search_web(query: str, max_results: int = 5) -> dict[str, Any]:
        return search_web(query, api_key=api_key, max_results=max_results)

    def _search_news(query: str, max_results: int = 5, days: int = 7) -> dict[str, Any]:
        return search_news(
            query,
            api_key=api_key,
            max_results=max_results,
            days=days,
        )

    return [
        AgentTool(
            name="search_web",
            description="General web search for background information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                    },
                },
                "required": ["query"],
            },
            handler=_search_web,
        ),
        AgentTool(
            name="search_news",
            description="Search recent news articles about companies or markets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "News search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Lookback window in days (default 7)",
                    },
                },
                "required": ["query"],
            },
            handler=_search_news,
        ),
    ]

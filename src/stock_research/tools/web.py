"""Web and news search via Tavily."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol


class TavilyLike(Protocol):
    def search(self, **kwargs: Any) -> dict[str, Any]:
        ...


def create_tavily_client(api_key: str) -> TavilyLike:
    """Build a Tavily client or raise if the API key is missing."""
    cleaned = api_key.strip()
    if not cleaned:
        raise ValueError(
            "TAVILY_API_KEY is required. Set it in .env to use web search."
        )
    from tavily import TavilyClient

    return TavilyClient(api_key=cleaned)


def _normalize_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        results.append(
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "content": item.get("content") or "",
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            }
        )
    return results


def search_web(
    query: str,
    *,
    api_key: str,
    max_results: int = 5,
    topic: str = "general",
    days: int | None = None,
    client: TavilyLike | None = None,
) -> dict[str, Any]:
    """Search the web with Tavily and return normalized results."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must be a non-empty string")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    tavily = client or create_tavily_client(api_key)
    kwargs: dict[str, Any] = {
        "query": cleaned,
        "max_results": max_results,
        "topic": topic,
        "search_depth": "basic",
        "include_answer": False,
    }
    if days is not None:
        kwargs["days"] = days

    payload = tavily.search(**kwargs)
    return {
        "query": cleaned,
        "topic": topic,
        "answer": payload.get("answer"),
        "results": _normalize_results(payload),
        "as_of": datetime.now(UTC).isoformat(),
        "source": "tavily",
    }


def search_news(
    query: str,
    *,
    api_key: str,
    max_results: int = 5,
    days: int = 7,
    client: TavilyLike | None = None,
) -> dict[str, Any]:
    """Search recent news articles for a query."""
    return search_web(
        query,
        api_key=api_key,
        max_results=max_results,
        topic="news",
        days=days,
        client=client,
    )

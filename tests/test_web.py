"""Tests for Tavily web search helpers."""

from __future__ import annotations

import pytest

from stock_research.tools import web


class FakeTavily:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": None,
            "results": [
                {
                    "title": "Apple reports earnings",
                    "url": "https://example.com/apple",
                    "content": "Apple beat estimates this quarter.",
                    "score": 0.91,
                    "published_date": "2026-09-27",
                }
            ],
        }


def test_search_web_normalizes_results():
    client = FakeTavily()
    payload = web.search_web(
        "Apple earnings",
        api_key="unused",
        max_results=3,
        client=client,
    )

    assert payload["query"] == "Apple earnings"
    assert payload["topic"] == "general"
    assert payload["source"] == "tavily"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "Apple reports earnings"
    assert client.calls[0]["max_results"] == 3
    assert client.calls[0]["topic"] == "general"


def test_search_news_uses_news_topic_and_days():
    client = FakeTavily()
    payload = web.search_news(
        "AAPL news",
        api_key="unused",
        max_results=2,
        days=3,
        client=client,
    )

    assert payload["topic"] == "news"
    assert client.calls[0]["topic"] == "news"
    assert client.calls[0]["days"] == 3
    assert client.calls[0]["max_results"] == 2


def test_create_tavily_client_requires_key():
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        web.create_tavily_client("  ")

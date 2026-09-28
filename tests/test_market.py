"""Tests for yfinance market tools."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_research.tools import market


class FakeFastInfo(dict):
    pass


class FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.fast_info = FakeFastInfo(
            {
                "last_price": 190.5,
                "previous_close": 188.0,
                "currency": "USD",
                "last_volume": 12_345_678,
            }
        )
        self.info = {
            "shortName": "Apple Inc.",
            "longName": "Apple Inc.",
            "symbol": symbol,
            "currency": "USD",
            "marketState": "REGULAR",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.5,
            "forwardPE": 28.1,
            "trailingEps": 6.2,
            "dividendYield": 0.0045,
            "fiftyTwoWeekHigh": 220.0,
            "fiftyTwoWeekLow": 160.0,
            "exchange": "NMS",
            "longBusinessSummary": "Apple designs consumer electronics.",
        }

    def history(self, period: str = "1mo", interval: str = "1d", auto_adjust: bool = True):
        idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
        return pd.DataFrame(
            {
                "Open": [180.0, 181.0],
                "High": [182.0, 183.0],
                "Low": [179.0, 180.5],
                "Close": [181.5, 182.25],
                "Volume": [1_000_000, 1_100_000],
            },
            index=idx,
        )


def test_normalize_ticker():
    assert market.normalize_ticker(" aapl ") == "AAPL"
    with pytest.raises(ValueError):
        market.normalize_ticker("  ")


def test_get_quote(monkeypatch):
    monkeypatch.setattr(market.yf, "Ticker", FakeTicker)
    quote = market.get_quote("aapl")
    assert quote["ticker"] == "AAPL"
    assert quote["price"] == 190.5
    assert quote["change"] == pytest.approx(2.5)
    assert quote["change_pct"] == pytest.approx((2.5 / 188.0) * 100)
    assert quote["currency"] == "USD"
    assert "as_of" in quote


def test_get_fundamentals(monkeypatch):
    monkeypatch.setattr(market.yf, "Ticker", FakeTicker)
    data = market.get_fundamentals("AAPL")
    assert data["company"] == "Apple Inc."
    assert data["sector"] == "Technology"
    assert data["market_cap"] == 3_000_000_000_000
    assert data["trailing_pe"] == 30.5


def test_get_history(monkeypatch):
    monkeypatch.setattr(market.yf, "Ticker", FakeTicker)
    rows = market.get_history("AAPL", period="5d")
    assert len(rows) == 2
    assert rows[0]["date"] == "2025-01-02"
    assert rows[1]["close"] == 182.25

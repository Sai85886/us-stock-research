"""Live US equity market data via yfinance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker must be a non-empty string")
    return cleaned


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def get_quote(ticker: str) -> dict[str, Any]:
    """Return a compact live quote for a ticker."""
    symbol = normalize_ticker(ticker)
    stock = yf.Ticker(symbol)
    fast = getattr(stock, "fast_info", {}) or {}
    info = stock.info or {}

    price = _safe_float(fast.get("last_price"))
    if price is None:
        price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    previous_close = _safe_float(fast.get("previous_close"))
    if previous_close is None:
        previous_close = _safe_float(info.get("previousClose") or info.get("regularMarketPreviousClose"))

    change = None
    change_pct = None
    if price is not None and previous_close not in (None, 0):
        change = price - previous_close
        change_pct = (change / previous_close) * 100

    currency = fast.get("currency") or info.get("currency")
    volume = _safe_int(fast.get("last_volume") or info.get("volume") or info.get("regularMarketVolume"))

    if price is None:
        raise ValueError(f"No quote data available for ticker: {symbol}")

    return {
        "ticker": symbol,
        "company": info.get("shortName") or info.get("longName") or symbol,
        "price": price,
        "previous_close": previous_close,
        "change": change,
        "change_pct": change_pct,
        "currency": currency,
        "volume": volume,
        "market_state": info.get("marketState"),
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
    }


def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Return key fundamentals and company profile fields."""
    symbol = normalize_ticker(ticker)
    info = yf.Ticker(symbol).info or {}

    if not info.get("shortName") and not info.get("longName") and not info.get("symbol"):
        # yfinance often returns a sparse dict for unknown tickers
        raise ValueError(f"No fundamentals available for ticker: {symbol}")

    return {
        "ticker": symbol,
        "company": info.get("shortName") or info.get("longName") or symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": _safe_int(info.get("marketCap")),
        "enterprise_value": _safe_int(info.get("enterpriseValue")),
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "trailing_eps": _safe_float(info.get("trailingEps")),
        "forward_eps": _safe_float(info.get("forwardEps")),
        "dividend_yield": _safe_float(info.get("dividendYield")),
        "profit_margins": _safe_float(info.get("profitMargins")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "earnings_growth": _safe_float(info.get("earningsGrowth")),
        "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
        "average_volume": _safe_int(info.get("averageVolume")),
        "currency": info.get("currency"),
        "exchange": info.get("exchange") or info.get("fullExchangeName"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
    }


def get_history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> list[dict[str, Any]]:
    """Return OHLCV history rows for a ticker."""
    symbol = normalize_ticker(ticker)
    frame = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if frame is None or frame.empty:
        raise ValueError(f"No history available for ticker: {symbol}")

    rows: list[dict[str, Any]] = []
    for timestamp, row in frame.iterrows():
        rows.append(
            {
                "date": timestamp.date().isoformat(),
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": _safe_int(row.get("Volume")),
            }
        )
    return rows

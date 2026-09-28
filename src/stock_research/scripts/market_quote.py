"""CLI for live yfinance market quotes and fundamentals."""

from __future__ import annotations

import argparse
import json
import sys

from stock_research.tools.market import get_fundamentals, get_history, get_quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live US equity market data via yfinance.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="Also print key fundamentals / profile fields",
    )
    parser.add_argument(
        "--history",
        metavar="PERIOD",
        default=None,
        help="Optional history period to print (e.g. 5d, 1mo, 3mo)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="History interval when --history is set (default: 1d)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )
    return parser.parse_args()


def _fmt_money(value: float | None, currency: str | None = None) -> str:
    if value is None:
        return "n/a"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def print_quote(quote: dict) -> None:
    print(
        f"{quote['ticker']} | {quote.get('company')}\n"
        f"  Price: {_fmt_money(quote.get('price'), quote.get('currency'))}\n"
        f"  Change: {_fmt_money(quote.get('change'))} ({_fmt_pct(quote.get('change_pct'))})\n"
        f"  Prev close: {_fmt_money(quote.get('previous_close'))}\n"
        f"  Volume: {_fmt_int(quote.get('volume'))}\n"
        f"  Market: {quote.get('market_state') or 'n/a'}\n"
        f"  As of: {quote.get('as_of')}"
    )


def _fmt_dividend_yield(value: float | None) -> str:
    """yfinance may return yield as a fraction or already in percent units."""
    if value is None:
        return "n/a"
    percent = value * 100 if value <= 0.1 else value
    return f"{percent:.2f}%"


def print_fundamentals(data: dict) -> None:
    print("\nFundamentals:")
    print(f"  Sector/Industry: {data.get('sector') or 'n/a'} / {data.get('industry') or 'n/a'}")
    print(f"  Market cap: {_fmt_int(data.get('market_cap'))}")
    print(f"  Trailing P/E: {data.get('trailing_pe') if data.get('trailing_pe') is not None else 'n/a'}")
    print(f"  Forward P/E: {data.get('forward_pe') if data.get('forward_pe') is not None else 'n/a'}")
    print(f"  EPS (ttm): {data.get('trailing_eps') if data.get('trailing_eps') is not None else 'n/a'}")
    print(f"  Dividend yield: {_fmt_dividend_yield(data.get('dividend_yield'))}")
    print(
        f"  52w range: {_fmt_money(data.get('fifty_two_week_low'))} - "
        f"{_fmt_money(data.get('fifty_two_week_high'))}"
    )
    print(f"  Exchange: {data.get('exchange') or 'n/a'}")
    summary = data.get("summary")
    if summary:
        print(f"  Summary: {str(summary)[:240].rstrip()}...")


def print_history(rows: list[dict]) -> None:
    print("\nHistory:")
    for row in rows[-10:]:
        print(
            f"  {row['date']}: close={_fmt_money(row.get('close'))} "
            f"volume={_fmt_int(row.get('volume'))}"
        )


def main() -> int:
    args = parse_args()

    try:
        quote = get_quote(args.ticker)
        payload: dict = {"quote": quote}

        fundamentals = None
        history = None
        if args.fundamentals:
            fundamentals = get_fundamentals(args.ticker)
            payload["fundamentals"] = fundamentals
        if args.history:
            history = get_history(args.ticker, period=args.history, interval=args.interval)
            payload["history"] = history
    except Exception as exc:  # noqa: BLE001 - CLI should show clean errors
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print_quote(quote)
    if fundamentals is not None:
        print_fundamentals(fundamentals)
    if history is not None:
        print_history(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download latest SEC 10-K filings for starter tickers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_research.config import get_settings
from stock_research.ingestion.edgar_client import EdgarClient

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download latest SEC 10-K filings.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=f"Tickers to download (default: {' '.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for downloaded filings (default: data/raw from config)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output_dir = args.output_dir or Path(settings.raw_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving filings to: {output_dir.resolve()}\n")

    downloaded = []
    errors: list[str] = []

    with EdgarClient(settings.sec_edgar_user_agent) as client:
        for ticker in args.tickers:
            try:
                metadata = client.download_latest_10k(ticker, output_dir)
                downloaded.append(metadata)
                print(
                    f"[OK] {metadata.ticker} | {metadata.company}\n"
                    f"     Form: {metadata.form} | Filed: {metadata.filing_date}\n"
                    f"     File: {metadata.local_path}\n"
                )
            except Exception as exc:  # noqa: BLE001 - CLI should continue on per-ticker failure
                errors.append(f"{ticker}: {exc}")
                print(f"[ERROR] {ticker}: {exc}\n", file=sys.stderr)

    print(f"Downloaded {len(downloaded)} of {len(args.tickers)} filings.")

    if errors:
        print("\nFailures:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

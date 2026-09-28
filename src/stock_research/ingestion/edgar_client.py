"""Client for fetching SEC EDGAR 10-K filings."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import httpx

SEC_DATA_BASE = "https://data.sec.gov"
SEC_WWW_BASE = "https://www.sec.gov"
TICKERS_URL = f"{SEC_WWW_BASE}/files/company_tickers.json"
REQUEST_DELAY_SECONDS = 0.2


@dataclass
class FilingMetadata:
    """Metadata for a downloaded SEC filing."""

    ticker: str
    company: str
    cik: str
    form: str
    filing_date: str
    accession_number: str
    primary_document: str
    source_url: str
    local_path: str


class EdgarClient:
    """Fetch company filings from SEC EDGAR."""

    def __init__(self, user_agent: str) -> None:
        if not user_agent.strip():
            raise ValueError(
                "SEC_EDGAR_USER_AGENT is required. "
                'Set it in .env, e.g. "Your Name your.email@example.com".'
            )
        self.user_agent = user_agent.strip()
        self._client = httpx.Client(
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=60.0,
            follow_redirects=True,
        )
        self._ticker_to_cik: dict[str, str] | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _throttle(self) -> None:
        time.sleep(REQUEST_DELAY_SECONDS)

    def _get_json(self, url: str) -> dict | list:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response.json()

    def _get_text(self, url: str) -> str:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response.text

    def _load_ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik

        payload = self._get_json(TICKERS_URL)
        ticker_map: dict[str, str] = {}
        for entry in payload.values():
            ticker = str(entry["ticker"]).upper()
            cik = str(entry["cik_str"]).zfill(10)
            ticker_map[ticker] = cik

        self._ticker_to_cik = ticker_map
        return ticker_map

    def resolve_cik(self, ticker: str) -> str:
        """Map a stock ticker to a zero-padded CIK."""
        normalized = ticker.upper().strip()
        ticker_map = self._load_ticker_map()
        if normalized not in ticker_map:
            raise ValueError(f"Unknown ticker: {ticker}")
        return ticker_map[normalized]

    def get_company_name(self, cik: str) -> str:
        """Return the company name from EDGAR submissions metadata."""
        submissions = self._get_submissions(cik)
        return str(submissions.get("name", "Unknown Company"))

    def _get_submissions(self, cik: str) -> dict:
        padded_cik = cik.zfill(10)
        url = f"{SEC_DATA_BASE}/submissions/CIK{padded_cik}.json"
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected submissions payload for CIK {cik}")
        return payload

    def find_latest_10k(self, ticker: str) -> dict:
        """Find metadata for the most recent 10-K filing."""
        cik = self.resolve_cik(ticker)
        submissions = self._get_submissions(cik)
        recent = submissions["filings"]["recent"]

        for index, form in enumerate(recent["form"]):
            if form == "10-K":
                return {
                    "ticker": ticker.upper(),
                    "company": str(submissions.get("name", "Unknown Company")),
                    "cik": cik,
                    "form": form,
                    "filing_date": recent["filingDate"][index],
                    "accession_number": recent["accessionNumber"][index],
                    "primary_document": recent["primaryDocument"][index],
                }

        raise ValueError(f"No 10-K filing found for {ticker}")

    def build_filing_url(self, cik: str, accession_number: str, primary_document: str) -> str:
        """Build the SEC Archives URL for a filing's primary document."""
        cik_path = str(int(cik))
        accession_path = accession_number.replace("-", "")
        return f"{SEC_WWW_BASE}/Archives/edgar/data/{cik_path}/{accession_path}/{primary_document}"

    def download_latest_10k(self, ticker: str, output_dir: Path) -> FilingMetadata:
        """Download the latest 10-K for a ticker into output_dir."""
        filing = self.find_latest_10k(ticker)
        source_url = self.build_filing_url(
            filing["cik"],
            filing["accession_number"],
            filing["primary_document"],
        )

        ticker_dir = output_dir / filing["ticker"]
        ticker_dir.mkdir(parents=True, exist_ok=True)

        extension = Path(filing["primary_document"]).suffix or ".html"
        filename = f"10-K_{filing['filing_date']}{extension}"
        local_path = ticker_dir / filename

        document = self._get_text(source_url)
        local_path.write_text(document, encoding="utf-8")

        metadata = FilingMetadata(
            ticker=filing["ticker"],
            company=filing["company"],
            cik=filing["cik"],
            form=filing["form"],
            filing_date=filing["filing_date"],
            accession_number=filing["accession_number"],
            primary_document=filing["primary_document"],
            source_url=source_url,
            local_path=str(local_path),
        )

        metadata_path = ticker_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(asdict(metadata), indent=2),
            encoding="utf-8",
        )

        return metadata
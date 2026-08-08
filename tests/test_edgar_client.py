from stock_research.ingestion.edgar_client import EdgarClient


def test_build_filing_url():
    client = EdgarClient("Test User test@example.com")
    url = client.build_filing_url(
        cik="0000320193",
        accession_number="0000320193-24-000123",
        primary_document="aapl-20240928.htm",
    )
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000123/aapl-20240928.htm"
    )


def test_resolve_cik_from_map():
    client = EdgarClient("Test User test@example.com")
    client._ticker_to_cik = {"AAPL": "0000320193", "MSFT": "0000789019"}
    assert client.resolve_cik("aapl") == "0000320193"

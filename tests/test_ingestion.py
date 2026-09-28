from stock_research.ingestion.chunker import chunk_filing
from stock_research.ingestion.loader import extract_text_from_html, split_into_sections
from stock_research.ingestion.models import FilingSection, LoadedFiling


def test_extract_text_from_html_removes_hidden_content():
    html = """
    <html><body>
      <div style="display:none">hidden metadata</div>
      <p>Apple designs consumer electronics.</p>
      <script>console.log("ignore")</script>
    </body></html>
    """
    text = extract_text_from_html(html)
    assert "Apple designs consumer electronics." in text
    assert "hidden metadata" not in text
    assert "console.log" not in text


def test_split_into_sections_detects_item_headers():
    text = """
    Cover page content
    Item 1. Business
    We design products worldwide.
    Item 1A. Risk Factors
    Competition is intense.
    Item 7. Management's Discussion and Analysis
    Revenue increased during the year.
    """
    sections = split_into_sections(text)
    names = [section.name for section in sections]
    assert "Preamble" in names
    assert "Item 1 - Business" in names
    assert "Item 1A - Risk Factors" in names
    assert "Item 7 - MD&A" in names


def test_chunk_filing_adds_metadata_and_overlap():
    filing = LoadedFiling(
        ticker="AAPL",
        company="Apple Inc.",
        cik="0000320193",
        form="10-K",
        filing_date="2025-10-31",
        source_file="data/raw/AAPL/10-K_2025-10-31.htm",
        full_text="example",
        sections=[
            FilingSection(
                name="Item 1A - Risk Factors",
                text="Risk sentence one. " * 80,
            )
        ],
    )

    chunks = chunk_filing(filing, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    assert chunks[0].ticker == "AAPL"
    assert chunks[0].section == "Item 1A - Risk Factors"
    assert chunks[0].chunk_id.startswith("AAPL_2025-10-31_")

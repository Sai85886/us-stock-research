"""Load SEC 10-K HTML filings into clean text."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from stock_research.ingestion.models import FilingSection, LoadedFiling

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTION_DEFINITIONS: list[tuple[str, re.Pattern[str]]] = [
    ("Item 1 - Business", re.compile(r"Item\s+1[\.\s]+Business", re.IGNORECASE)),
    ("Item 1A - Risk Factors", re.compile(r"Item\s+1A[\.\s]+Risk Factors", re.IGNORECASE)),
    (
        "Item 1B - Unresolved Staff Comments",
        re.compile(r"Item\s+1B[\.\s]+Unresolved Staff Comments", re.IGNORECASE),
    ),
    ("Item 1C - Cybersecurity", re.compile(r"Item\s+1C[\.\s]+Cybersecurity", re.IGNORECASE)),
    ("Item 2 - Properties", re.compile(r"Item\s+2[\.\s]+Properties", re.IGNORECASE)),
    (
        "Item 3 - Legal Proceedings",
        re.compile(r"Item\s+3[\.\s]+Legal Proceedings", re.IGNORECASE),
    ),
    (
        "Item 7 - MD&A",
        re.compile(
            r"Item\s+7[\.\s]+Management['\u2019]?s Discussion and Analysis",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 7A - Market Risk",
        re.compile(
            r"Item\s+7A[\.\s]+Quantitative and Qualitative Disclosures About Market Risk",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 8 - Financial Statements",
        re.compile(r"Item\s+8[\.\s]+Financial Statements", re.IGNORECASE),
    ),
]

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def load_metadata(metadata_path: Path) -> dict:
    """Load filing metadata written during download."""
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def extract_text_from_html(html: str) -> str:
    """Extract readable text from SEC inline XBRL HTML."""
    soup = BeautifulSoup(html, "lxml")

    for element in soup.find_all(["script", "style", "noscript"]):
        element.decompose()

    for hidden in soup.find_all("div", style=True):
        style = hidden.get("style", "")
        if isinstance(style, str) and "display:none" in style.replace(" ", "").lower():
            hidden.decompose()

    for hidden in soup.find_all("ix:header"):
        hidden.decompose()

    body = soup.body
    if isinstance(body, Tag):
        text = body.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def split_into_sections(text: str) -> list[FilingSection]:
    """Split filing text into labeled 10-K sections when headers are found."""
    matches: list[tuple[int, str]] = []

    for section_name, pattern in SECTION_DEFINITIONS:
        for match in pattern.finditer(text):
            matches.append((match.start(), section_name))

    if not matches:
        return [FilingSection(name="Full Document", text=text)]

    matches.sort(key=lambda item: item[0])

    # Keep the last occurrence of each section header. The first match is often
    # in the table of contents; the last match is usually the real section body.
    last_match_by_name: dict[str, int] = {}
    for position, name in matches:
        last_match_by_name[name] = position

    deduped = [(position, name) for name, position in last_match_by_name.items()]
    deduped.sort(key=lambda item: item[0])

    sections: list[FilingSection] = []

    if deduped[0][0] > 0:
        preamble = text[: deduped[0][0]].strip()
        if preamble:
            sections.append(FilingSection(name="Preamble", text=preamble))

    # The enumerate function in Python returns both the index and the item from the iterable.
    # Here, it allows us to access both the position in 'deduped' (index) and the actual (start, name) tuple.
    for index, (start, name) in enumerate(deduped):
        end = deduped[index + 1][0] if index + 1 < len(deduped) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append(FilingSection(name=name, text=section_text))

    return sections


def load_filing(filing_path: Path, metadata: dict) -> LoadedFiling:
    """Load a filing HTML file and return structured text."""
    html = filing_path.read_text(encoding="utf-8", errors="ignore")
    full_text = extract_text_from_html(html)
    sections = split_into_sections(full_text)

    return LoadedFiling(
        ticker=metadata["ticker"],
        company=metadata["company"],
        cik=metadata["cik"],
        form=metadata["form"],
        filing_date=metadata["filing_date"],
        source_file=str(filing_path),
        full_text=full_text,
        sections=sections,
    )


def discover_filings(raw_data_dir: Path) -> list[tuple[Path, Path]]:
    """Find metadata.json files and their corresponding filing paths."""
    filings: list[tuple[Path, Path]] = []

    for metadata_path in sorted(raw_data_dir.glob("*/metadata.json")):
        metadata = load_metadata(metadata_path)
        filing_path = Path(metadata["local_path"])
        if not filing_path.is_absolute():
            filing_path = Path.cwd() / filing_path
        if not filing_path.exists():
            raise FileNotFoundError(f"Filing not found for {metadata['ticker']}: {filing_path}")
        filings.append((metadata_path, filing_path))

    return filings

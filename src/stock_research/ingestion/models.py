"""Shared data models for ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FilingSection:
    """A labeled section extracted from a filing."""

    name: str
    text: str


@dataclass
class LoadedFiling:
    """Parsed filing text with section boundaries."""

    ticker: str
    company: str
    cik: str
    form: str
    filing_date: str
    source_file: str
    full_text: str
    sections: list[FilingSection]


@dataclass
class DocumentChunk:
    """A chunk of filing text ready for embedding."""

    chunk_id: str
    chunk_index: int
    text: str
    ticker: str
    company: str
    cik: str
    form: str
    filing_date: str
    section: str
    source_file: str

    def to_dict(self) -> dict:
        return asdict(self)

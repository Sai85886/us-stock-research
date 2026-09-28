"""Split parsed filings into retrieval-friendly chunks."""

from __future__ import annotations

from stock_research.ingestion.models import DocumentChunk, LoadedFiling

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively split text using progressively finer separators."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    separator_found = False

    for separator in SEPARATORS:
        if separator == "":
            break
        if separator not in text:
            continue

        separator_found = True
        parts = text.split(separator)
        current = ""

        for part in parts:
            candidate = part if not current else current + separator + part
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(_split_text(current, chunk_size, chunk_overlap))
            current = part

        if current:
            chunks.extend(_split_text(current, chunk_size, chunk_overlap))
        break

    if not separator_found:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    return _apply_overlap(chunks, chunk_overlap)


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    """Add overlap between consecutive chunks."""
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        previous = overlapped[-1]
        prefix = previous[-chunk_overlap:] if len(previous) > chunk_overlap else previous
        overlapped.append(prefix + chunk)

    return overlapped


def chunk_filing(
    filing: LoadedFiling,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split a loaded filing into metadata-rich chunks."""
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section in filing.sections:
        section_chunks = _split_text(section.text, chunk_size, chunk_overlap)
        for section_chunk in section_chunks:
            chunk_id = f"{filing.ticker}_{filing.filing_date}_{chunk_index:04d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=section_chunk,
                    ticker=filing.ticker,
                    company=filing.company,
                    cik=filing.cik,
                    form=filing.form,
                    filing_date=filing.filing_date,
                    section=section.name,
                    source_file=filing.source_file,
                )
            )
            chunk_index += 1

    return chunks

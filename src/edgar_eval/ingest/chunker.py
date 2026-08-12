"""Turn sectionised elements into retrievable chunks.

Two decisions carry most of the retrieval quality:

**Chunk within a section, never across one.** `chunk_by_title` is applied to
each Item's elements separately rather than to the whole filing. Run over the
whole document it will happily merge the tail of Item 1A into the head of Item
1B when both are short, and a chunk that spans two Items cannot be filtered to
either -- which defeats the metadata filtering that the hybrid query is built
around.

**Embed a contextual header, store the bare text.** A chunk reading "increased
8% year over year" names neither the company, the period, nor the metric, so no
query retrieves it. Prepending a deterministic header before embedding fixes
that at zero inference cost. The header is *not* stored in `text`, so it never
leaks into a quoted answer, and re-embedding later does not require re-chunking.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from edgar_eval.config import settings
from edgar_eval.ingest.sections import FormType, ItemRef

# Rough token estimate. Deliberately not tiktoken: that is OpenAI's tokenizer
# and would be wrong for bge-m3's XLM-R vocabulary anyway. This value only
# drives reporting and a sanity bound, never a truncation decision -- bge-m3's
# 8192-token window is far past anything chunk_max_characters can produce.
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """One retrievable unit, ready to be embedded and written."""

    text: str
    embed_text: str
    item: ItemRef | None
    section_path: list[str]
    chunk_index: int
    element_index: int
    element_types: list[str] = field(default_factory=list)
    table_html: str | None = None
    page_number: int | None = None

    @property
    def has_table(self) -> bool:
        return self.table_html is not None

    @property
    def token_count(self) -> int:
        return max(1, len(self.embed_text) // _CHARS_PER_TOKEN)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def build_context_header(
    *,
    company_name: str,
    ticker: str,
    form_type: FormType,
    fiscal_year: int,
    fiscal_period: str,
    item: ItemRef | None,
    section_path: list[str] | None = None,
) -> str:
    """The deterministic preamble prepended to a chunk before embedding.

    Format is stable across the corpus so it contributes the same lexical
    signal everywhere:

        Apple Inc. (AAPL) · 10-K · FY2023 · Item 7 — Management's Discussion...
        Section: Results of Operations > Segment Operating Performance
    """
    period = "FY" if fiscal_period == "FY" else fiscal_period
    parts = [f"{company_name} ({ticker})", form_type, f"{period}{fiscal_year}"]
    if item is not None:
        parts.append(f"Item {item.number} — {item.title(form_type)}")
    header = " · ".join(parts)

    if section_path:
        header += "\nSection: " + " > ".join(section_path)
    return header


def _clean(text: str) -> str:
    """Collapse the whitespace noise that HTML filings are full of."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_table(html: str | None, text: str) -> str:
    """Render a table for *embedding*, not for display.

    Raw HTML embeds badly -- tag soup dominates the token budget and the
    numbers get lost. `unstructured` already gives us a flattened text form;
    we keep the HTML separately for the generator and the UI, which is where
    structure actually helps a reader.
    """
    return _clean(text)


def chunk_section(
    elements: list[Any],
    *,
    item: ItemRef | None,
    start_chunk_index: int,
    company_name: str,
    ticker: str,
    form_type: FormType,
    fiscal_year: int,
    fiscal_period: str,
) -> list[Chunk]:
    """Chunk one Item's elements. See module docstring for why per-section."""
    if not elements:
        return []

    from unstructured.chunking.title import chunk_by_title

    chunked = chunk_by_title(
        elements,
        max_characters=settings.chunk_max_characters,
        new_after_n_chars=settings.chunk_new_after_n_chars,
        combine_text_under_n_chars=settings.chunk_combine_under_n_chars,
        overlap=settings.chunk_overlap,
        multipage_sections=True,
    )

    chunks: list[Chunk] = []
    for offset, element in enumerate(chunked):
        metadata = getattr(element, "metadata", None)
        table_html = getattr(metadata, "text_as_html", None)
        orig_types = list(getattr(metadata, "orig_elements", None) or [])
        element_types = sorted({type(e).__name__ for e in orig_types}) or [type(element).__name__]

        text = flatten_table(table_html, str(element)) if table_html else _clean(str(element))
        if not text:
            continue

        section_path = [t for t in (getattr(metadata, "section", None),) if t]
        header = build_context_header(
            company_name=company_name,
            ticker=ticker,
            form_type=form_type,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            item=item,
            section_path=section_path,
        )

        chunks.append(
            Chunk(
                text=text,
                embed_text=f"{header}\n\n{text}",
                item=item,
                section_path=section_path,
                chunk_index=start_chunk_index + offset,
                element_index=getattr(metadata, "element_index", start_chunk_index + offset),
                element_types=element_types,
                table_html=table_html,
                page_number=getattr(metadata, "page_number", None),
            )
        )
    return chunks

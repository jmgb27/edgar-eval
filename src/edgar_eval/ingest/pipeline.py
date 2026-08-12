"""Fetch -> partition -> sectionise -> chunk -> embed -> write.

Chunks are embedded and written in batches rather than all at once. On the
reference hardware the embedder manages ~1.5 chunks/s, so a 10-K is minutes of
work; batching means progress is visible, memory stays flat, and an interrupted
run leaves a partially-populated corpus that a re-run converges rather than
duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.ingest.chunker import Chunk, chunk_section
from edgar_eval.ingest.edgar import EdgarClient, Filing
from edgar_eval.ingest.partition import PartitionedElement, Partitioner, UnstructuredHtmlPartitioner
from edgar_eval.ingest.sections import ItemRef, sectionise
from edgar_eval.ingest.writer import delete_chunks, upsert_filing, write_chunks
from edgar_eval.logging import get_logger

log = get_logger(__name__)

EMBED_BATCH = 16


@dataclass
class IngestResult:
    filing: Filing
    elements: int
    chunks: int
    written: int
    items: list[str]


def _group_by_item(
    elements: list[PartitionedElement],
    items: list[ItemRef | None],
) -> list[tuple[ItemRef | None, list[PartitionedElement]]]:
    """Split the element stream into contiguous runs sharing an Item.

    Chunking happens per run, never across runs -- that is what guarantees no
    chunk spans two Items.
    """
    groups: list[tuple[ItemRef | None, list[PartitionedElement]]] = []
    current_item: ItemRef | None = None
    current: list[PartitionedElement] = []

    for element, item in zip(elements, items, strict=True):
        if item != current_item:
            if current:
                groups.append((current_item, current))
            current_item, current = item, []
        current.append(element)
    if current:
        groups.append((current_item, current))
    return groups


def chunk_filing(
    filing: Filing,
    elements: list[PartitionedElement],
) -> list[Chunk]:
    assignment = sectionise(
        [e.text for e in elements],
        form_type=filing.form_type,
        categories=[e.category for e in elements],
    )

    chunks: list[Chunk] = []
    for item, group in _group_by_item(elements, assignment.items):
        # Elements before Item 1 are the cover page: legally required, and
        # genuinely useful (it carries the company name, exchange and share
        # count), so it is kept rather than dropped.
        raw_elements = [e.raw for e in group if e.raw is not None]
        if not raw_elements:
            continue
        chunks.extend(
            chunk_section(
                raw_elements,
                item=item,
                start_chunk_index=len(chunks),
                company_name=filing.company_name,
                ticker=filing.ticker,
                form_type=filing.form_type,
                fiscal_year=filing.fiscal_year,
                fiscal_period=filing.fiscal_period,
            )
        )
    return chunks


def ingest_filing(
    filing: Filing,
    *,
    edgar: EdgarClient,
    embedder: EmbeddingsClient,
    partitioner: Partitioner | None = None,
    replace: bool = True,
) -> IngestResult:
    partitioner = partitioner or UnstructuredHtmlPartitioner()

    log.info(
        "ingest.start",
        accession=filing.accession_no,
        ticker=filing.ticker,
        form=filing.form_type,
        fy=filing.fiscal_year,
    )

    html = edgar.fetch_document(filing)
    elements = partitioner.partition(html)
    chunks = chunk_filing(filing, elements)

    upsert_filing(filing)
    if replace:
        # See writer.delete_chunks: without this, re-ingesting under different
        # chunk settings leaves both generations in the table.
        removed = delete_chunks(filing.accession_no)
        if removed:
            log.info("ingest.replaced", accession=filing.accession_no, removed=removed)

    written = 0
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors = embedder.embed([c.embed_text for c in batch], kind="passage")
        written += write_chunks(filing, batch, vectors)
        log.info(
            "ingest.progress",
            accession=filing.accession_no,
            done=min(start + EMBED_BATCH, len(chunks)),
            total=len(chunks),
        )

    items = sorted({c.item.key for c in chunks if c.item})
    log.info(
        "ingest.done",
        accession=filing.accession_no,
        elements=len(elements),
        chunks=len(chunks),
        written=written,
        items=len(items),
    )
    return IngestResult(
        filing=filing, elements=len(elements), chunks=len(chunks), written=written, items=items
    )

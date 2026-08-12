"""Write filings and chunks to Postgres.

Idempotent by construction: `filings` is keyed on the SEC accession number and
`chunks` carries a UNIQUE (accession_no, content_sha256). Re-running an ingest
therefore converges rather than duplicating the corpus, which matters because
the ablation study re-ingests the same filings under different chunking
settings all day.
"""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb  # noqa: F401  (kept for future metadata column)

from edgar_eval.config import settings
from edgar_eval.db import connection
from edgar_eval.ingest.chunker import Chunk
from edgar_eval.ingest.edgar import Filing
from edgar_eval.logging import get_logger

log = get_logger(__name__)

_UPSERT_FILING = """
INSERT INTO filings (
    accession_no, cik, ticker, company_name, form_type, fiscal_year,
    fiscal_period, filing_date, period_end_date, source_url, ingest_version
) VALUES (
    %(accession_no)s, %(cik)s, %(ticker)s, %(company_name)s, %(form_type)s,
    %(fiscal_year)s, %(fiscal_period)s, %(filing_date)s, %(period_end_date)s,
    %(source_url)s, %(ingest_version)s
)
ON CONFLICT (accession_no) DO UPDATE SET
    company_name   = EXCLUDED.company_name,
    source_url     = EXCLUDED.source_url,
    ingest_version = EXCLUDED.ingest_version,
    ingested_at    = now()
"""

_INSERT_CHUNK = """
INSERT INTO chunks (
    accession_no, ticker, form_type, fiscal_year, fiscal_period, period_end_date,
    item, item_title, section_path, text, embed_text, table_html, has_table,
    element_types, chunk_index, element_index, page_number, source_url,
    token_count, content_sha256, embedding
) VALUES (
    %(accession_no)s, %(ticker)s, %(form_type)s, %(fiscal_year)s, %(fiscal_period)s,
    %(period_end_date)s, %(item)s, %(item_title)s, %(section_path)s, %(text)s,
    %(embed_text)s, %(table_html)s, %(has_table)s, %(element_types)s,
    %(chunk_index)s, %(element_index)s, %(page_number)s, %(source_url)s,
    %(token_count)s, %(content_sha256)s, %(embedding)s
)
ON CONFLICT (accession_no, content_sha256) DO NOTHING
"""


def upsert_filing(filing: Filing) -> None:
    with connection() as conn:
        conn.execute(
            _UPSERT_FILING,
            {
                "accession_no": filing.accession_no,
                "cik": filing.cik,
                "ticker": filing.ticker,
                "company_name": filing.company_name,
                "form_type": filing.form_type,
                "fiscal_year": filing.fiscal_year,
                "fiscal_period": filing.fiscal_period,
                "filing_date": filing.filing_date,
                "period_end_date": filing.period_end_date,
                "source_url": filing.source_url,
                "ingest_version": settings.ingest_version,
            },
        )
        conn.commit()


def delete_chunks(accession_no: str) -> int:
    """Drop a filing's chunks so it can be re-ingested under new settings.

    Needed because the UNIQUE constraint makes re-ingest a no-op otherwise:
    changing chunk_max_characters produces *different* chunks, and without a
    delete the table would end up holding both generations at once, silently
    doubling the corpus and corrupting the ablation.
    """
    with connection() as conn:
        cur = conn.execute("DELETE FROM chunks WHERE accession_no = %s", (accession_no,))
        conn.commit()
        return cur.rowcount


def write_chunks(filing: Filing, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """Insert chunks with their embeddings. Returns rows actually written."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunk/embedding length mismatch: {len(chunks)} vs {len(embeddings)}")
    if not chunks:
        return 0

    rows: list[dict[str, Any]] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        rows.append(
            {
                "accession_no": filing.accession_no,
                "ticker": filing.ticker,
                "form_type": filing.form_type,
                "fiscal_year": filing.fiscal_year,
                "fiscal_period": filing.fiscal_period,
                "period_end_date": filing.period_end_date,
                "item": chunk.item.number if chunk.item else None,
                "item_title": chunk.item.title(filing.form_type) if chunk.item else None,
                "section_path": chunk.section_path,
                "text": chunk.text,
                "embed_text": chunk.embed_text,
                "table_html": chunk.table_html,
                "has_table": chunk.has_table,
                "element_types": chunk.element_types,
                "chunk_index": chunk.chunk_index,
                "element_index": chunk.element_index,
                "page_number": chunk.page_number,
                "source_url": filing.source_url,
                "token_count": chunk.token_count,
                "content_sha256": chunk.content_sha256,
                "embedding": embedding,
            }
        )

    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(_INSERT_CHUNK, rows)
        conn.commit()

    log.info("writer.chunks", accession=filing.accession_no, written=len(rows))
    return len(rows)


def corpus_bounds() -> dict[str, Any]:
    """What the corpus actually contains.

    Used to clamp model-extracted filters: the single most common retrieval
    failure is a question about "2024" against a corpus ending in 2023, where
    an unclamped filter returns nothing and the agent concludes the answer is
    absent rather than that its filter was wrong.
    """
    with connection() as conn:
        row = conn.execute(
            """
            SELECT min(fiscal_year) AS year_min,
                   max(fiscal_year) AS year_max,
                   array_agg(DISTINCT ticker)    AS tickers,
                   array_agg(DISTINCT form_type) AS form_types,
                   count(*)                      AS filings
            FROM filings
            """
        ).fetchone()
    return dict(row) if row else {}

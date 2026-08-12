"""Hybrid retrieval: the RRF query plus the session settings it depends on."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from edgar_eval.config import REPO_ROOT, settings
from edgar_eval.db import connection
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.logging import get_logger
from edgar_eval.retrieve.filters import Filters

log = get_logger(__name__)

QUERY_PATH = REPO_ROOT / "db" / "queries" / "hybrid_rrf.sql"


@lru_cache(maxsize=1)
def _sql() -> str:
    return QUERY_PATH.read_text()


@dataclass
class Candidate:
    id: int
    text: str
    embed_text: str
    ticker: str
    form_type: str
    fiscal_year: int
    fiscal_period: str
    item: str | None
    item_title: str | None
    accession_no: str
    source_url: str
    rrf_score: float
    in_dense: bool
    in_lexical: bool
    has_table: bool = False
    table_html: str | None = None
    section_path: list[str] = None  # type: ignore[assignment]
    page_number: int | None = None
    rerank_score: float | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Candidate:
        return cls(
            id=row["id"],
            text=row["text"],
            embed_text=row["embed_text"],
            ticker=row["ticker"],
            form_type=row["form_type"],
            fiscal_year=row["fiscal_year"],
            fiscal_period=row["fiscal_period"],
            item=row["item"],
            item_title=row["item_title"],
            accession_no=row["accession_no"],
            source_url=row["source_url"],
            rrf_score=float(row["rrf_score"]),
            in_dense=bool(row["in_dense"]),
            in_lexical=bool(row["in_lexical"]),
            has_table=bool(row["has_table"]),
            table_html=row["table_html"],
            section_path=list(row["section_path"] or []),
            page_number=row["page_number"],
        )

    @property
    def citation(self) -> str:
        item = f" Item {self.item}" if self.item else ""
        return f"{self.ticker} {self.form_type} FY{self.fiscal_year}{item}"


def hybrid_search(
    question: str,
    *,
    filters: Filters | None = None,
    embedder: EmbeddingsClient,
    pool: int | None = None,
    fused: int | None = None,
) -> list[Candidate]:
    """Dense + lexical retrieval fused by RRF, filtered on metadata."""
    filters = filters or Filters()
    qvec = embedder.embed_one(question, kind="query")

    params = {
        "qvec": str(qvec),
        "qtext": question,
        "pool": pool or settings.retrieval_pool,
        "fused": fused or settings.retrieval_fused,
        "rrf_k": settings.rrf_k,
        **filters.as_params(),
    }

    with connection() as conn, conn.cursor() as cur:
        # Session-scoped, not global: these govern this query's recall/latency
        # tradeoff and should not leak to other statements on a pooled
        # connection.
        #
        # iterative_scan is the load-bearing one. Without it a filtered vector
        # search examines the first ef_search candidates *before* applying the
        # WHERE clause, so a selective filter (one ticker, one Item) can return
        # a handful of rows out of a corpus that holds hundreds of matches.
        # 'relaxed_order' rather than 'strict_order' because everything here is
        # reranked downstream anyway, so approximate ordering costs nothing.
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(settings.hnsw_ef_search)}")
        cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        cur.execute(_sql(), params)
        rows = cur.fetchall()

    candidates = [Candidate.from_row(row) for row in rows]
    log.debug(
        "retrieve.hybrid",
        question=question[:80],
        returned=len(candidates),
        dense_only=sum(1 for c in candidates if c.in_dense and not c.in_lexical),
        lexical_only=sum(1 for c in candidates if c.in_lexical and not c.in_dense),
        both=sum(1 for c in candidates if c.in_dense and c.in_lexical),
    )
    return candidates


def rerank_candidates(
    question: str,
    candidates: list[Candidate],
    *,
    embedder: EmbeddingsClient,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[Candidate]:
    """Cross-encoder rerank, then threshold.

    Candidates below `min_score` are dropped rather than padded out to `top_k`.
    Padding would hand the generator plausible-looking but irrelevant context,
    which is precisely how a grounded-sounding wrong answer gets produced.
    """
    if not candidates:
        return []
    top_k = top_k or settings.retrieval_topk
    min_score = settings.rerank_min_score if min_score is None else min_score

    # embed_text, not text. The cross-encoder must see the same contextual
    # header the embedder saw -- company, fiscal year, Item -- or short
    # table chunks are scored as bare grids of numbers. Apple's effective
    # tax rate lives in a 157-character chunk reading only
    #   "2023 2022 2021 Provision for income taxes ... Effective tax rate 14.7 %"
    # which, stripped of context, loses to the cash-flow statement on a
    # question that names neither.
    docs = [(c.embed_text if settings.rerank_with_context else c.text) for c in candidates]
    hits = embedder.rerank(question, docs)
    ranked: list[Candidate] = []
    for hit in hits:
        candidate = candidates[hit.index]
        candidate.rerank_score = hit.score
        if hit.score >= min_score:
            ranked.append(candidate)

    log.debug(
        "retrieve.rerank",
        considered=len(candidates),
        above_threshold=len(ranked),
        kept=min(len(ranked), top_k),
    )
    return ranked[:top_k]

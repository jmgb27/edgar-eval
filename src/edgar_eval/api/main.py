"""FastAPI service.

`/search` is the endpoint that needs no credentials: hybrid retrieval,
reranking and filtering are all local. It is also the UI's default tab, because
it is the surface where the engineering is visible -- you can watch the
reranker reorder results and see which arm found each one.

`/ask` synthesises an answer when a generation model is configured and returns
the top-ranked passage verbatim when one is not.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from edgar_eval import observability as obs
from edgar_eval.config import settings
from edgar_eval.db import close_pool, connection
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.graph.build import build_graph
from edgar_eval.ingest.writer import corpus_bounds
from edgar_eval.logging import configure_logging, get_logger
from edgar_eval.retrieve.filters import Filters, clamp_to_corpus
from edgar_eval.retrieve.hybrid import Candidate, hybrid_search, rerank_candidates

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    obs.verify_auth()  # warns and disables on failure; never blocks startup
    _state["embedder"] = EmbeddingsClient()
    _state["graph"] = build_graph(embedder=_state["embedder"])
    log.info(
        "api.ready",
        llm=settings.qwen_model if settings.llm_configured else "extractive",
        tracing=obs.langfuse_enabled(),
    )
    yield
    obs.flush()
    _state["embedder"].close()
    close_pool()


app = FastAPI(title="edgar-eval", version="0.1.0", lifespan=lifespan)


# ── schemas ─────────────────────────────────────────────────
class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    tickers: list[str] | None = None
    forms: list[str] | None = None
    items: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    rerank: bool = True


class Passage(BaseModel):
    id: int
    text: str
    table_html: str | None
    has_table: bool
    ticker: str
    form_type: str
    fiscal_year: int
    item: str | None
    item_title: str | None
    source_url: str
    rrf_score: float
    rerank_score: float | None
    # Provenance: which arm found this. Rendered as a badge in the UI so the
    # hybrid is visible rather than asserted.
    in_dense: bool
    in_lexical: bool


class SearchResponse(BaseModel):
    question: str
    filters: dict[str, Any]
    filter_notes: list[str]
    passages: list[Passage]
    stats: dict[str, int]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]]
    groundedness: float
    extractive: bool
    trace_notes: list[str]
    trace_id: str | None
    session_id: str


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int = Field(ge=-1, le=1)
    comment: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)
    trace_id: str | None = None


def _to_passage(c: Candidate) -> Passage:
    return Passage(
        id=c.id,
        text=c.text,
        table_html=c.table_html,
        has_table=c.has_table,
        ticker=c.ticker,
        form_type=c.form_type,
        fiscal_year=c.fiscal_year,
        item=c.item,
        item_title=c.item_title,
        source_url=c.source_url,
        rrf_score=c.rrf_score,
        rerank_score=c.rerank_score,
        in_dense=c.in_dense,
        in_lexical=c.in_lexical,
    )


# ── endpoints ───────────────────────────────────────────────
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        with connection() as conn:
            row = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
        chunks = int(row["n"]) if row else 0
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc

    return {
        "status": "ok",
        "chunks": chunks,
        "corpus": corpus_bounds(),
        "generation_model": settings.qwen_model if settings.llm_configured else None,
        "extractive_mode": not settings.llm_configured,
        "tracing": obs.langfuse_enabled(),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Retrieval only. Needs no API key of any kind."""
    embedder: EmbeddingsClient = _state["embedder"]
    clamped = clamp_to_corpus(
        Filters(
            tickers=req.tickers,
            forms=req.forms,
            items=req.items,
            year_min=req.year_min,
            year_max=req.year_max,
        )
    )
    candidates = hybrid_search(req.question, filters=clamped, embedder=embedder)
    passages = (
        rerank_candidates(req.question, candidates, embedder=embedder)
        if req.rerank
        else candidates[: settings.retrieval_topk]
    )

    return SearchResponse(
        question=req.question,
        filters=clamped.as_params(),
        filter_notes=clamped.notes,
        passages=[_to_passage(c) for c in passages],
        stats={
            "fused": len(candidates),
            "returned": len(passages),
            "dense_only": sum(1 for c in candidates if c.in_dense and not c.in_lexical),
            "lexical_only": sum(1 for c in candidates if c.in_lexical and not c.in_dense),
            "both": sum(1 for c in candidates if c.in_dense and c.in_lexical),
        },
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    session_id = req.session_id or str(uuid.uuid4())
    graph = _state["graph"]

    with obs.trace_context(
        trace_name="edgar-eval-answer",
        session_id=session_id,
        tags=["ask", "extractive" if not settings.llm_configured else "generative"],
        metadata={"question": req.question},
    ):
        result = graph.invoke(
            {"question": req.question, "session_id": session_id, "trace_notes": []},
            config={
                "callbacks": obs.run_callbacks(),
                "configurable": {"thread_id": session_id},
            },
        )
        trace_id = obs.current_trace_id()
        obs.score("groundedness-selfcheck", float(result.get("groundedness", 1.0)))
        obs.score("n-contexts", float(len(result.get("contexts") or [])))
        obs.score("retrieval-attempts", float(result.get("attempts", 0)))

    return AskResponse(
        question=req.question,
        answer=result.get("answer", ""),
        citations=list(result.get("citations") or []),
        groundedness=float(result.get("groundedness", 1.0)),
        extractive=bool(result.get("extractive", not settings.llm_configured)),
        trace_notes=list(result.get("trace_notes") or []),
        trace_id=trace_id,
        session_id=session_id,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Thumbs from the UI.

    Written to Postgres *and* forwarded to Langfuse. Local storage matters
    because tracing is off by default, and feedback that only exists in a
    vendor nobody configured is feedback that does not exist.
    """
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (trace_id, session_id, question, answer, rating, comment, chunk_ids)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                req.trace_id,
                req.session_id,
                req.question,
                req.answer,
                req.rating,
                req.comment,
                req.chunk_ids,
            ),
        )
        conn.commit()
    obs.score("user-feedback", float(req.rating), comment=req.comment)
    return {"status": "recorded"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

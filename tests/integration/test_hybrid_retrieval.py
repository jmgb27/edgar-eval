"""Hybrid retrieval against a real ingested corpus.

Requires Postgres with at least one filing ingested and the embeddings service
running:

    make up && uv run python scripts/ingest_cli.py --ticker AAPL --years 2023 2023
"""

from __future__ import annotations

import httpx
import pytest

from edgar_eval.db import connection
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.retrieve.filters import Filters, clamp_to_corpus
from edgar_eval.retrieve.hybrid import hybrid_search, rerank_candidates

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder() -> EmbeddingsClient:
    c = EmbeddingsClient()
    try:
        c.health()
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        pytest.skip(f"embeddings service unavailable: {exc}")
    return c


@pytest.fixture(scope="module", autouse=True)
def _require_corpus() -> None:
    try:
        with connection() as conn:
            row = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"database unavailable: {exc}")
    if not row or row["n"] == 0:
        pytest.skip("no chunks ingested; run scripts/ingest_cli.py first")


def test_lexical_arm_contributes(embedder: EmbeddingsClient) -> None:
    """Regression: the lexical arm must actually return rows.

    `websearch_to_tsquery` AND-joins its lexemes, so the original query matched
    zero chunks for "net cash provided by operating activities" -- Apple writes
    "generated", not "provided". That silently degraded hybrid retrieval to
    dense-only while every test still passed and the benchmark still produced
    numbers. The arm now ORs; this asserts it stays that way.
    """
    candidates = hybrid_search(
        "What was net cash provided by operating activities?",
        embedder=embedder,
    )
    assert candidates
    assert any(c.in_lexical for c in candidates), (
        "lexical arm returned nothing -- hybrid retrieval has silently become dense-only"
    )


def test_proper_nouns_surface_lexical_only_hits(embedder: EmbeddingsClient) -> None:
    """The case hybrid exists for: an exact product name the embedder blurs
    into its neighbours but the lexical index matches precisely."""
    candidates = hybrid_search("Apple Watch", embedder=embedder)
    assert any(c.in_lexical and not c.in_dense for c in candidates), (
        "expected at least one lexical-only hit for a proper noun"
    )


def test_reranker_surfaces_the_answering_table(embedder: EmbeddingsClient) -> None:
    """End-to-end: the effective tax rate lives in a table, and the table has
    to come back first."""
    candidates = hybrid_search("effective tax rate", embedder=embedder)
    top = rerank_candidates("effective tax rate", candidates, embedder=embedder)

    assert top, "everything was filtered out by the rerank threshold"
    assert "effective tax rate" in top[0].text.lower()
    assert top[0].rerank_score is not None and top[0].rerank_score > 0.5


def test_item_filter_restricts_results(embedder: EmbeddingsClient) -> None:
    candidates = hybrid_search(
        "What risks does the company face?",
        filters=Filters(items=["1A"]),
        embedder=embedder,
    )
    assert candidates
    assert {c.item for c in candidates} == {"1A"}


def test_rerank_scores_descend(embedder: EmbeddingsClient) -> None:
    candidates = hybrid_search("revenue by segment", embedder=embedder)
    top = rerank_candidates("revenue by segment", candidates, embedder=embedder)
    scores = [c.rerank_score for c in top]
    assert scores == sorted(scores, reverse=True)


def test_out_of_range_year_is_clamped_not_emptied() -> None:
    """A question about a year after the corpus must retrieve the closest year
    with an explanation, not return nothing and imply the fact does not exist."""
    clamped = clamp_to_corpus(Filters(year_min=2099, year_max=2099))
    assert clamped.year_min != 2099
    assert clamped.notes, "the clamp must be reported so the answer can mention it"


def test_unknown_ticker_is_dropped_rather_than_clamped() -> None:
    """Clamping a ticker would be a guess. Answering about the wrong company is
    worse than saying the company is absent."""
    clamped = clamp_to_corpus(Filters(tickers=["NOTREAL"]))
    assert clamped.tickers is None
    assert any("NOTREAL" in note for note in clamped.notes)


def test_reranker_does_not_rank_an_unrelated_table_first(embedder: EmbeddingsClient) -> None:
    """Regression: the cash-flow statement used to win a tax-rate question.

    The reranker was scoring the bare chunk text while the embedder scored the
    text *plus* its contextual header. Stripped of that header, a 157-character
    tax table is a bare grid of numbers, and lost to the much larger cash-flow
    grid on a question naming neither. Whatever the ablation eventually decides
    about `rerank_with_context`, a passage containing no tax content at all must
    not be the top hit for a tax question.
    """
    candidates = hybrid_search("What was the effective tax rate?", embedder=embedder)
    top = rerank_candidates("What was the effective tax rate?", candidates, embedder=embedder)

    assert top, "everything was filtered out by the rerank threshold"
    winner = top[0].text.lower()
    assert "tax" in winner, f"top hit mentions no tax at all: {top[0].text[:120]!r}"


def test_rerank_input_follows_the_setting(embedder: EmbeddingsClient) -> None:
    """`rerank_with_context` is an ablation dimension; both paths must work."""
    from edgar_eval.config import settings

    candidates = hybrid_search("effective tax rate", embedder=embedder)
    original = settings.rerank_with_context
    try:
        for value in (True, False):
            settings.rerank_with_context = value
            top = rerank_candidates("effective tax rate", candidates, embedder=embedder)
            assert top, f"no results with rerank_with_context={value}"
    finally:
        settings.rerank_with_context = original

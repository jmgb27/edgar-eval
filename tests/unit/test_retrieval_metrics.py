"""Retrieval metrics.

These run with no database, no embeddings service and no API key, which is the
whole point: the Recall@k column of the README table has to be reproducible by
someone who has cloned the repo and set nothing.
"""

from __future__ import annotations

from edgar_eval.eval.retrieval_metrics import (
    aggregate,
    chunk_matches_span,
    ndcg_at_k,
    normalise,
    recall_at_k,
    reciprocal_rank,
    score_question,
)


# ── span matching ───────────────────────────────────────────
def test_normalise_collapses_filing_whitespace() -> None:
    """Gold spans are copied from rendered pages; chunks come from HTML. The
    two differ in whitespace far more often than in words."""
    assert normalise("Effective  tax\nrate\xa014.7 %") == "effective tax rate 14.7 %"


def test_span_matching_survives_reflow() -> None:
    chunk = "Provision for income taxes $ 16,741\n\nEffective  tax rate   14.7 %"
    assert chunk_matches_span(chunk, "Effective tax rate 14.7 %")


def test_span_matching_is_not_merely_topical() -> None:
    """A chunk about tax that lacks the figure is not a hit. A looser rule
    would report healthy recall for a system that never surfaces the number."""
    chunk = "The Company is subject to changes in tax rates in various jurisdictions."
    assert not chunk_matches_span(chunk, "Effective tax rate 14.7 %")


# ── per-question metrics ────────────────────────────────────
def _result(ranks: list[int], n_gold: int = 1):  # type: ignore[no-untyped-def]
    retrieved = [(i, f"chunk {i}") for i in range(1, 11)]
    r = score_question(question_id="q1", category="single_fact", gold_spans=[], retrieved=retrieved)
    r.relevant_ranks = ranks
    r.n_gold = n_gold
    return r


def test_recall_counts_gold_spans_not_retrieved_chunks() -> None:
    """Denominator is the number of gold spans: a question needing three
    supporting passages is not fully recalled by finding one."""
    assert recall_at_k(_result([1], n_gold=3), 10) == 1 / 3
    assert recall_at_k(_result([1, 4, 9], n_gold=3), 10) == 1.0


def test_recall_respects_the_cutoff() -> None:
    assert recall_at_k(_result([9], n_gold=1), 5) == 0.0
    assert recall_at_k(_result([9], n_gold=1), 10) == 1.0


def test_reciprocal_rank_uses_the_best_hit() -> None:
    assert reciprocal_rank(_result([3, 7])) == 1 / 3
    assert reciprocal_rank(_result([])) == 0.0


def test_ndcg_rewards_ordering_where_recall_cannot() -> None:
    """The reranker's entire job is ordering, and recall is blind to it."""
    first = ndcg_at_k(_result([1], n_gold=1), 10)
    tenth = ndcg_at_k(_result([10], n_gold=1), 10)
    assert first == 1.0
    assert tenth < first
    assert recall_at_k(_result([1], n_gold=1), 10) == recall_at_k(_result([10], n_gold=1), 10)


# ── scoring against real text ───────────────────────────────
def test_score_question_finds_the_rank_of_each_span() -> None:
    retrieved = [
        (10, "Mine safety disclosures. Not applicable."),
        (20, "Effective tax rate 14.7 %"),
        (30, "Net sales by segment"),
    ]
    result = score_question(
        question_id="q1",
        category="table_lookup",
        gold_spans=["Effective tax rate 14.7 %"],
        retrieved=retrieved,
    )
    assert result.relevant_ranks == [2]
    assert result.hit


def test_a_repeated_span_cannot_inflate_recall() -> None:
    """Boilerplate repeated across chunks would otherwise push recall past 1."""
    retrieved = [(1, "Not applicable."), (2, "Not applicable."), (3, "Not applicable.")]
    result = score_question(
        question_id="q1",
        category="single_fact",
        gold_spans=["Not applicable."],
        retrieved=retrieved,
    )
    assert result.relevant_ranks == [1]
    assert recall_at_k(result, 10) == 1.0


def test_missing_span_is_a_clean_miss() -> None:
    result = score_question(
        question_id="q1",
        category="single_fact",
        gold_spans=["a figure that is not there"],
        retrieved=[(1, "something else entirely")],
    )
    assert not result.hit
    assert recall_at_k(result, 10) == 0.0


# ── aggregation ─────────────────────────────────────────────
def test_unanswerable_questions_are_excluded_from_retrieval_metrics() -> None:
    """There is no span to find, so including them would drag recall down for
    behaving correctly."""
    answerable = score_question(
        question_id="a", category="single_fact", gold_spans=["x"], retrieved=[(1, "x")]
    )
    refusal = score_question(
        question_id="b",
        category="unanswerable",
        gold_spans=[],
        retrieved=[(2, "irrelevant")],
        unanswerable=True,
        abstained=True,
    )
    summary = aggregate([answerable, refusal], k=10)

    assert summary["n"] == 2
    assert summary["n_answerable"] == 1
    assert summary["recall@10"] == 1.0
    assert summary["abstain_recall"] == 1.0


def test_abstain_recall_penalises_a_confident_answer() -> None:
    refusals = [
        score_question(
            question_id=str(i),
            category="unanswerable",
            gold_spans=[],
            retrieved=[],
            unanswerable=True,
            abstained=abstained,
        )
        for i, abstained in enumerate([True, True, False, True])
    ]
    assert aggregate(refusals)["abstain_recall"] == 0.75


def test_per_category_breakdown_is_reported() -> None:
    """Tables are the hard case and get their own floor in the CI gate, so the
    breakdown has to exist before the threshold can reference it."""
    results = [
        score_question(
            question_id="1", category="table_lookup", gold_spans=["x"], retrieved=[(1, "x")]
        ),
        score_question(
            question_id="2", category="single_fact", gold_spans=["y"], retrieved=[(1, "z")]
        ),
    ]
    summary = aggregate(results)
    assert summary["by_category"]["table_lookup"]["recall@10"] == 1.0
    assert summary["by_category"]["single_fact"]["recall@10"] == 0.0


def test_empty_input_does_not_divide_by_zero() -> None:
    summary = aggregate([])
    assert summary["n"] == 0
    assert summary["recall@10"] == 0.0
    assert summary["abstain_recall"] is None

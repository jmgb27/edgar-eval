"""Retrieval metrics, computed in-repo from gold spans.

Deliberately judge-free and therefore free, deterministic, and runnable with no
API key. That is what lets `make eval-retrieval` reproduce the Recall@k column
of the README table on a stranger's laptop, and it is why the retrieval half of
the benchmark is not hostage to an LLM judge's run-to-run noise.

Relevance is decided by *span containment*: a retrieved chunk is relevant if it
contains one of the question's gold spans, normalised for whitespace. That is a
strict definition -- a chunk about the right topic that lacks the span does not
count -- which is the point. A metric that rewarded topical proximity would
report a healthy Recall@10 for a system that never actually surfaces the number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

_WS = re.compile(r"\s+")
# Filings are typeset, gold spans are usually copied from a rendered page,
# and the two disagree about quote characters far more often than about
# words. Apple writes "the Company\u2019s total net sales" with U+2019; a span
# typed with an ASCII apostrophe would silently never match, and the
# question would score as a retrieval miss rather than as a bad span.
_QUOTES = str.maketrans(
    {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


def normalise(text: str) -> str:
    """Collapse whitespace and case so span matching survives HTML reflow.

    Filings are full of non-breaking spaces and line breaks inserted by the
    renderer rather than the author, and a gold span copied from a rendered
    page will not be byte-identical to the chunk text.
    """
    return _WS.sub(" ", text.replace("\xa0", " ").translate(_QUOTES)).strip().lower()


def chunk_matches_span(chunk_text: str, span: str) -> bool:
    return normalise(span) in normalise(chunk_text)


@dataclass
class QuestionResult:
    question_id: str
    category: str
    retrieved_ids: list[int]
    relevant_ranks: list[int] = field(default_factory=list)
    n_gold: int = 0
    unanswerable: bool = False
    abstained: bool | None = None

    @property
    def hit(self) -> bool:
        return bool(self.relevant_ranks)


def recall_at_k(result: QuestionResult, k: int) -> float:
    """Fraction of gold spans found within the top k.

    Denominator is the number of gold spans, not the number retrieved, so a
    question with three supporting spans is only fully recalled when all three
    surface.
    """
    if result.n_gold == 0:
        return 0.0
    found = sum(1 for rank in result.relevant_ranks if rank <= k)
    return min(1.0, found / result.n_gold)


def reciprocal_rank(result: QuestionResult) -> float:
    return 1.0 / min(result.relevant_ranks) if result.relevant_ranks else 0.0


def ndcg_at_k(result: QuestionResult, k: int) -> float:
    """Binary-gain nDCG@k.

    Included alongside recall because recall is blind to ordering: a system
    that buries the answer at rank 10 scores the same Recall@10 as one that
    puts it first, and the reranker's entire job is that ordering.
    """
    if result.n_gold == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in result.relevant_ranks if rank <= k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(result.n_gold, k)))
    return dcg / ideal if ideal else 0.0


def score_question(
    *,
    question_id: str,
    category: str,
    gold_spans: list[str],
    retrieved: list[tuple[int, str]],
    unanswerable: bool = False,
    abstained: bool | None = None,
) -> QuestionResult:
    """Rank the retrieved chunks against a question's gold spans.

    `retrieved` is [(chunk_id, chunk_text)] in rank order, rank 1 first.
    """
    result = QuestionResult(
        question_id=question_id,
        category=category,
        retrieved_ids=[cid for cid, _ in retrieved],
        n_gold=len(gold_spans),
        unanswerable=unanswerable,
        abstained=abstained,
    )
    # A span counts once, at the best rank where it appears -- otherwise a
    # boilerplate span repeated across chunks would inflate recall past 1.
    for span in gold_spans:
        for rank, (_cid, text) in enumerate(retrieved, start=1):
            if chunk_matches_span(text, span):
                result.relevant_ranks.append(rank)
                break
    result.relevant_ranks.sort()
    return result


def aggregate(results: list[QuestionResult], *, k: int = 10) -> dict[str, Any]:
    """Corpus-level metrics, plus a per-category breakdown.

    Unanswerable questions are excluded from retrieval metrics -- there is no
    span to find -- and scored separately as abstention recall, because a system
    that answers them confidently is failing in the way that matters most for a
    financial corpus.
    """
    answerable = [r for r in results if not r.unanswerable]
    unanswerable = [r for r in results if r.unanswerable]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, Any] = {
        "n": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        f"recall@{k}": round(mean([recall_at_k(r, k) for r in answerable]), 4),
        "mrr": round(mean([reciprocal_rank(r) for r in answerable]), 4),
        f"ndcg@{k}": round(mean([ndcg_at_k(r, k) for r in answerable]), 4),
        "hit_rate": round(mean([1.0 if r.hit else 0.0 for r in answerable]), 4),
    }

    scored = [r for r in unanswerable if r.abstained is not None]
    summary["abstain_recall"] = (
        round(mean([1.0 if r.abstained else 0.0 for r in scored]), 4) if scored else None
    )

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({r.category for r in answerable}):
        subset = [r for r in answerable if r.category == category]
        by_category[category] = {
            "n": len(subset),
            f"recall@{k}": round(mean([recall_at_k(r, k) for r in subset]), 4),
            f"ndcg@{k}": round(mean([ndcg_at_k(r, k) for r in subset]), 4),
        }
    summary["by_category"] = by_category
    return summary


# ── citation accuracy ───────────────────────────────────────
#
# Judge-free, deterministic, and free. It answers the question an LLM judge is
# worst at: not "does the answer read as supported" but "does it point at the
# right document".
#
# A right answer with an invented citation is the single worst failure mode in a
# financial corpus -- it is confidently wrong *and* it defeats the verification
# the citation exists to enable. LLM judges are weak here precisely because a
# fabricated citation looks exactly like a real one.
#
# This needs no generator: the top-ranked passage carries its own accession
# number and Item, so the check is a set comparison against the gold set.


@dataclass
class CitationResult:
    question_id: str
    accession_correct: bool | None  # None when the gold set pins no accession
    item_correct: bool | None
    cited_accession: str | None
    cited_item: str | None


def score_citation(
    *,
    question_id: str,
    expected_accession: str | None,
    expected_items: list[str] | None,
    cited_accession: str | None,
    cited_item: str | None,
) -> CitationResult:
    """Did the top passage come from the filing and section the gold set names?

    `expected_accession` is None for cross-filing questions, where no single
    accession is correct; that dimension is then reported as unscored rather
    than as a failure, because scoring it either way would be a lie.
    """
    accession_correct = (
        None if expected_accession is None else cited_accession == expected_accession
    )
    item_correct = None if not expected_items else cited_item in set(expected_items)
    return CitationResult(
        question_id=question_id,
        accession_correct=accession_correct,
        item_correct=item_correct,
        cited_accession=cited_accession,
        cited_item=cited_item,
    )


def aggregate_citations(results: list[CitationResult]) -> dict[str, Any]:
    scored_accession = [r for r in results if r.accession_correct is not None]
    scored_item = [r for r in results if r.item_correct is not None]

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "n_scored_accession": len(scored_accession),
        "n_scored_item": len(scored_item),
        "accession_accuracy": rate([bool(r.accession_correct) for r in scored_accession]),
        "item_accuracy": rate([bool(r.item_correct) for r in scored_item]),
    }

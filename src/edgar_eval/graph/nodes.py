"""Graph nodes.

Every node appends one line to `trace_notes`, so the UI can render what the
agent did without a reviewer having to open Langfuse.

Two merges are deliberate and worth defending:

  * `analyze_query` does decomposition, filter extraction and scope
    classification in one structured-output call. They are one decision
    producing one object; splitting them costs two round trips and adds a
    second failure mode for no benefit.

  * `grade_context` grades the retrieved set in one call rather than
    per-document. Per-document grading is the textbook Corrective-RAG shape,
    but it is 6 model calls for a decision that branches three ways.
"""

from __future__ import annotations

import json
import re
from typing import Any

from edgar_eval.config import settings
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.graph import prompts
from edgar_eval.graph.state import Citation, RagState
from edgar_eval.llm import build_llm, llm_configured
from edgar_eval.logging import get_logger
from edgar_eval.retrieve.filters import Filters, clamp_to_corpus
from edgar_eval.retrieve.hybrid import Candidate, hybrid_search, rerank_candidates

log = get_logger(__name__)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_YEAR_RE = re.compile(r"\b(?:FY\s?)?(19|20)(\d{2})\b", re.IGNORECASE)
_ITEM_HINTS = {
    "risk factor": "1A",
    "risk": "1A",
    "md&a": "7",
    "management's discussion": "7",
    "properties": "2",
    "legal proceeding": "3",
    "financial statement": "8",
    "controls and procedures": "9A",
    "cybersecurity": "1C",
    "business": "1",
}


def _json_from(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that a bare `json.loads`
    is not a safe contract, and a parse failure here should degrade to defaults
    rather than fail the request.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def extract_filters_heuristically(question: str) -> Filters:
    """Rule-based filter extraction, used when no model is configured.

    Deliberately conservative: it only picks up things stated outright. A
    heuristic that guesses is worse than one that abstains, because a wrong
    filter empties the result set.
    """
    filters = Filters()

    years = [int(f"{c}{y}") for c, y in _YEAR_RE.findall(question)]
    if years:
        filters.year_min, filters.year_max = min(years), max(years)

    lowered = question.lower()
    for hint, item in _ITEM_HINTS.items():
        if hint in lowered:
            filters.items = [item]
            break

    if "10-q" in lowered or "quarterly" in lowered:
        filters.forms = ["10-Q"]
    elif "10-k" in lowered or "annual report" in lowered:
        filters.forms = ["10-K"]

    # Uppercase tokens that are not sentence-initial words. Common English
    # words in caps would produce phantom tickers, so a stoplist is applied and
    # the corpus clamp drops anything that survives but is not held.
    stop = {"I", "A", "THE", "SEC", "US", "USA", "CEO", "CFO", "GAAP", "FY", "Q1", "Q2", "Q3", "Q4"}
    tickers = [t for t in _TICKER_RE.findall(question) if t not in stop and len(t) >= 2]
    if tickers:
        filters.tickers = tickers
    return filters


# ── nodes ───────────────────────────────────────────────────
def analyze_query(state: RagState) -> RagState:
    question = state["question"]

    if not llm_configured():
        filters = extract_filters_heuristically(question)
        clamped = clamp_to_corpus(filters)
        return {
            "subqueries": [question],
            "filters": clamped.as_params(),
            "filter_notes": clamped.notes,
            "scope": "in_scope",
            "attempts": 0,
            "regenerations": 0,
            "extractive": True,
            "trace_notes": ["analyze: extractive mode (no generation model configured)"],
        }

    llm = build_llm(max_tokens=800)
    response = llm.invoke(
        [
            ("system", prompts.ANALYZE_SYSTEM),
            ("human", f"Question: {question}\n\nReturn JSON only."),
        ]
    )
    parsed = _json_from(str(response.content))

    filters = Filters(
        tickers=parsed.get("tickers") or None,
        forms=parsed.get("forms") or None,
        items=parsed.get("items") or None,
        year_min=parsed.get("year_min"),
        year_max=parsed.get("year_max"),
    )
    clamped = clamp_to_corpus(filters)
    subqueries = [q for q in (parsed.get("subqueries") or []) if isinstance(q, str)] or [question]

    note = f"analyze: {len(subqueries)} subquery(ies), filters={clamped.as_params()}"
    if clamped.notes:
        note += f" ({'; '.join(clamped.notes)})"

    return {
        "subqueries": subqueries[:3],
        "filters": clamped.as_params(),
        "filter_notes": clamped.notes,
        "scope": parsed.get("scope", "in_scope"),
        "attempts": 0,
        "regenerations": 0,
        "extractive": False,
        "trace_notes": [note],
    }


def hybrid_retrieve(state: RagState, *, embedder: EmbeddingsClient) -> RagState:
    filters = Filters(**state["filters"])
    merged: dict[int, Candidate] = {}

    for subquery in state.get("subqueries") or [state["question"]]:
        for candidate in hybrid_search(subquery, filters=filters, embedder=embedder):
            # Keep the best RRF score across subqueries rather than summing:
            # summing rewards a chunk merely for matching several subqueries,
            # which over-favours generic passages.
            existing = merged.get(candidate.id)
            if existing is None or candidate.rrf_score > existing.rrf_score:
                merged[candidate.id] = candidate

    candidates = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    candidates = candidates[: settings.retrieval_fused]

    dense_only = sum(1 for c in candidates if c.in_dense and not c.in_lexical)
    lexical_only = sum(1 for c in candidates if c.in_lexical and not c.in_dense)
    return {
        "candidates": candidates,
        "trace_notes": [
            f"retrieve: {len(candidates)} candidates "
            f"({dense_only} dense-only, {lexical_only} lexical-only)"
        ],
    }


def rerank(state: RagState, *, embedder: EmbeddingsClient) -> RagState:
    contexts = rerank_candidates(
        state["question"], state.get("candidates") or [], embedder=embedder
    )
    top = f"{contexts[0].rerank_score:.2f}" if contexts else "n/a"
    return {
        "contexts": contexts,
        "trace_notes": [f"rerank: kept {len(contexts)} above threshold (top score {top})"],
    }


def grade_context(state: RagState) -> RagState:
    contexts = state.get("contexts") or []

    if not contexts:
        return {
            "grade": "insufficient",
            "grade_reason": "nothing survived the rerank threshold",
            "trace_notes": ["grade: insufficient (no contexts)"],
        }

    if not llm_configured():
        # Without a model there is nothing to grade with; the reranker's own
        # threshold is the only signal available, and it already passed.
        return {
            "grade": "sufficient",
            "grade_reason": "extractive mode: reranker threshold is the only signal",
            "trace_notes": ["grade: skipped (extractive mode)"],
        }

    passages = "\n\n".join(
        f"[S{i + 1}] ({c.citation}) {c.text[:1200]}" for i, c in enumerate(contexts)
    )
    llm = build_llm(max_tokens=400)
    response = llm.invoke(
        [
            ("system", prompts.GRADE_SYSTEM),
            ("human", f"Question: {state['question']}\n\nPassages:\n{passages}\n\nJSON only."),
        ]
    )
    parsed = _json_from(str(response.content))
    grade = parsed.get("grade", "partial")
    reason = parsed.get("reason", "")
    return {
        "grade": grade if grade in {"sufficient", "partial", "insufficient"} else "partial",
        "grade_reason": reason,
        "trace_notes": [f"grade: {grade} — {reason}"],
    }


def rewrite_query(state: RagState) -> RagState:
    attempts = state.get("attempts", 0) + 1
    if not llm_configured():
        return {"attempts": attempts, "trace_notes": ["rewrite: skipped (extractive mode)"]}

    contexts = state.get("contexts") or []
    retrieved = "\n".join(f"- ({c.citation}) {c.text[:200]}" for c in contexts[:4]) or "(nothing)"

    llm = build_llm(max_tokens=300)
    response = llm.invoke(
        [
            ("system", prompts.REWRITE_SYSTEM),
            (
                "human",
                f"Question: {state['question']}\n"
                f"Shortfall: {state.get('grade_reason', 'unknown')}\n"
                f"Retrieved:\n{retrieved}\n\nJSON only.",
            ),
        ]
    )
    rewritten = _json_from(str(response.content)).get("query") or state["question"]

    # Widen the filters on a retry: an over-narrow Item or year filter is the
    # most common reason a first attempt comes back thin.
    filters = dict(state["filters"])
    widened = []
    if filters.get("items"):
        filters["items"] = None
        widened.append("items")
    if filters.get("forms"):
        filters["forms"] = None
        widened.append("forms")

    note = f"rewrite (attempt {attempts}): {rewritten!r}"
    if widened:
        note += f" — dropped filters: {', '.join(widened)}"
    return {
        "subqueries": [rewritten],
        "filters": filters,
        "attempts": attempts,
        "trace_notes": [note],
    }


def _citations_for(contexts: list[Candidate]) -> list[Citation]:
    return [
        Citation(
            marker=f"S{i + 1}",
            chunk_id=c.id,
            ticker=c.ticker,
            form_type=c.form_type,
            fiscal_year=c.fiscal_year,
            item=c.item,
            item_title=c.item_title,
            source_url=c.source_url,
            quote=c.text[:400],
        )
        for i, c in enumerate(contexts)
    ]


def generate(state: RagState) -> RagState:
    contexts = state.get("contexts") or []
    citations = _citations_for(contexts)

    if not llm_configured():
        # Extractive mode: return the top passage verbatim rather than
        # synthesising. Honest about what it is; the UI banners it.
        top = contexts[0]
        return {
            "answer": top.text,
            "citations": citations[:1],
            "groundedness": 1.0,
            "trace_notes": ["generate: extractive — top-ranked passage returned verbatim"],
        }

    passages = "\n\n".join(
        f"[S{i + 1}] ({c.citation})\n{c.table_html or c.text}" for i, c in enumerate(contexts)
    )
    notes = state.get("filter_notes") or []
    caveat = f"\n\nNote for your answer: {' '.join(notes)}" if notes else ""

    llm = build_llm(max_tokens=1500)
    response = llm.invoke(
        [
            ("system", prompts.GENERATE_SYSTEM),
            ("human", f"Question: {state['question']}\n\nPassages:\n{passages}{caveat}"),
        ]
    )
    answer = str(response.content).strip()
    cited = set(re.findall(r"\[S\d+\]", answer))
    return {
        "answer": answer,
        "citations": citations,
        "trace_notes": [f"generate: {len(answer)} chars, {len(cited)} distinct citation markers"],
    }


def verify_groundedness(state: RagState) -> RagState:
    if not llm_configured() or not state.get("answer"):
        return {"groundedness": 1.0, "unsupported_claims": [], "trace_notes": []}

    contexts = state.get("contexts") or []
    passages = "\n\n".join(f"[S{i + 1}] {c.text[:1000]}" for i, c in enumerate(contexts))

    llm = build_llm(max_tokens=600)
    response = llm.invoke(
        [
            ("system", prompts.VERIFY_SYSTEM),
            ("human", f"Answer:\n{state['answer']}\n\nPassages:\n{passages}\n\nJSON only."),
        ]
    )
    parsed = _json_from(str(response.content))
    try:
        groundedness = float(parsed.get("groundedness", 1.0))
    except (TypeError, ValueError):
        groundedness = 1.0
    unsupported = [c for c in (parsed.get("unsupported_claims") or []) if isinstance(c, str)]

    return {
        "groundedness": max(0.0, min(1.0, groundedness)),
        "unsupported_claims": unsupported,
        "trace_notes": [
            f"verify: groundedness {groundedness:.2f}, {len(unsupported)} unsupported claim(s)"
        ],
    }


def abstain(state: RagState) -> RagState:
    notes = state.get("filter_notes") or []
    reason = state.get("grade_reason") or "the corpus does not contain passages answering this"

    lines = [f"I can't answer that from this corpus. {reason}."]
    if notes:
        lines.append(" ".join(notes))
    if state.get("scope") == "out_of_scope":
        lines = [
            "That question can't be answered from SEC filings — it asks for a prediction "
            "or advice rather than a disclosed fact."
        ]

    return {
        "answer": " ".join(lines),
        "citations": [],
        "groundedness": 1.0,
        "trace_notes": ["abstain"],
    }

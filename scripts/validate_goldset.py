"""Validate the gold set against the corpus.

Every `reference_contexts` span must appear verbatim (modulo whitespace) in the
chunk it claims to come from. This is what makes a drafted gold set *grounded*
rather than plausible: a question whose gold span cannot be located in the
corpus is an invented fact, and an invented fact in the gold set silently
corrupts every number in the README.

Run before every eval, and in CI.
"""

from __future__ import annotations

import argparse
import json
import sys

from edgar_eval.config import REPO_ROOT
from edgar_eval.db import close_pool, connection
from edgar_eval.eval.retrieval_metrics import chunk_matches_span

REQUIRED = ("id", "category", "question", "reference", "reference_contexts")
CATEGORIES = {
    "single_fact",
    "table_lookup",
    "multi_filing_comparison",
    "temporal",
    "risk_factor_synthesis",
    "unanswerable",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="curated")
    args = parser.parse_args()

    path = REPO_ROOT / "eval" / "golden" / f"{args.set}.jsonl"
    questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    problems: list[str] = []
    seen_ids: set[str] = set()

    with connection() as conn:
        rows = conn.execute("SELECT id, ticker, item, text FROM chunks").fetchall()
    chunks = {r["id"]: r for r in rows}

    for q in questions:
        qid = q.get("id", "<missing id>")
        for field in REQUIRED:
            if field not in q:
                problems.append(f"{qid}: missing field {field!r}")
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id")
        seen_ids.add(qid)

        if q.get("category") not in CATEGORIES:
            problems.append(f"{qid}: unknown category {q.get('category')!r}")

        unanswerable = q.get("category") == "unanswerable"
        spans = q.get("reference_contexts") or []

        if unanswerable:
            if spans:
                problems.append(f"{qid}: unanswerable questions must have no gold spans")
            continue
        if not spans:
            problems.append(f"{qid}: answerable question has no gold spans")
            continue

        source_ids = q.get("source_chunk_ids") or []
        for span in spans:
            # Prefer the declared source chunk, but accept the span appearing
            # anywhere -- chunk ids shift when chunking settings change, and the
            # span is the real claim.
            in_declared = any(
                cid in chunks and chunk_matches_span(chunks[cid]["text"], span)
                for cid in source_ids
            )
            anywhere = [cid for cid, row in chunks.items() if chunk_matches_span(row["text"], span)]
            if not anywhere:
                problems.append(f"{qid}: span not found anywhere in the corpus: {span!r}")
            elif not in_declared and source_ids:
                problems.append(
                    f"{qid}: span not in declared chunk(s) {source_ids}, "
                    f"but found in {anywhere[:3]}: {span!r}"
                )

    close_pool()

    counts: dict[str, int] = {}
    for q in questions:
        counts[q.get("category", "?")] = counts.get(q.get("category", "?"), 0) + 1

    print(f"\n  {len(questions)} questions in {path.name}")
    for category, n in sorted(counts.items()):
        print(f"    {category:26s} {n}")

    if problems:
        print(f"\n  {len(problems)} problem(s):")
        for p in problems:
            print(f"    ✗ {p}")
        return 1

    print("\n  every gold span located in the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

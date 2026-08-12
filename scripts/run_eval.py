"""Run the curated evaluation set.

Two modes, and the split is the point:

    uv run python scripts/run_eval.py --retrieval-only    # no API key, free
    uv run python scripts/run_eval.py                     # + Ragas, needs a judge

`--retrieval-only` computes Recall@k, MRR and nDCG from the gold spans using
plain code. It is deterministic, costs nothing, and reproduces the retrieval
columns of the README table on a machine with no credentials at all. That is
what makes half the benchmark verifiable by a stranger.

The generation metrics need a judge, and a judge is a model with run-to-run
noise and a per-run cost, so they are opt-in and reported separately.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from edgar_eval.config import REPO_ROOT, settings
from edgar_eval.db import close_pool
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.eval.retrieval_metrics import aggregate, score_question
from edgar_eval.graph.prompts import prompt_fingerprint
from edgar_eval.logging import configure_logging, get_logger
from edgar_eval.retrieve.filters import Filters, clamp_to_corpus
from edgar_eval.retrieve.hybrid import hybrid_search, rerank_candidates

log = get_logger(__name__)

GOLDEN_DIR = REPO_ROOT / "eval" / "golden"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"gold set not found: {path}")
    questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not questions:
        raise SystemExit(f"gold set is empty: {path}")
    return questions


def _git_sha() -> str:
    try:
        # Full path: ruff S607 rightly objects to resolving `git` off PATH
        # in something whose output is stamped onto published results.
        return subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def provenance(*, retrieval_only: bool, gold_path: Path) -> dict[str, Any]:
    """Everything needed to attribute a change in the numbers.

    Without this a regression between two runs cannot be pinned on a code
    change rather than a prompt edit, a model swap or a different gold set --
    and an unattributable regression is indistinguishable from noise.
    """
    import hashlib

    return {
        "git_sha": _git_sha(),
        "gold_set": gold_path.name,
        "gold_set_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest()[:16],
        "prompt_fingerprint": prompt_fingerprint(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "rerank_with_context": settings.rerank_with_context,
        "retrieval": {
            "pool": settings.retrieval_pool,
            "fused": settings.retrieval_fused,
            "topk": settings.retrieval_topk,
            "rrf_k": settings.rrf_k,
            "rerank_min_score": settings.rerank_min_score,
            "hnsw_ef_search": settings.hnsw_ef_search,
        },
        "judge_model": None if retrieval_only else settings.judge_model,
        "judge_temperature": None if retrieval_only else 0.0,
        "generation_model": None if retrieval_only else settings.qwen_model,
        "python": platform.python_version(),
    }


def evaluate_retrieval(
    questions: list[dict[str, Any]], *, embedder: EmbeddingsClient, k: int
) -> tuple[list[Any], list[dict[str, Any]]]:
    results, per_question = [], []

    for q in questions:
        filters = clamp_to_corpus(
            Filters(
                tickers=q.get("tickers"),
                forms=q.get("forms"),
                items=q.get("items"),
                year_min=q.get("year_min"),
                year_max=q.get("year_max"),
            )
        )
        started = time.monotonic()
        candidates = hybrid_search(q["question"], filters=filters, embedder=embedder)
        ranked = rerank_candidates(
            q["question"], candidates, embedder=embedder, top_k=max(k, settings.retrieval_topk)
        )
        latency = time.monotonic() - started

        unanswerable = q.get("category") == "unanswerable"
        result = score_question(
            question_id=q["id"],
            category=q.get("category", "uncategorised"),
            gold_spans=q.get("reference_contexts", []),
            retrieved=[(c.id, c.text) for c in ranked],
            unanswerable=unanswerable,
            # In retrieval-only mode there is no generated answer, so abstention
            # is approximated by "nothing survived the rerank threshold". The
            # generative run measures the real thing.
            abstained=(not ranked) if unanswerable else None,
        )
        results.append(result)
        per_question.append(
            {
                "id": q["id"],
                "category": result.category,
                "question": q["question"],
                "hit": result.hit,
                "best_rank": min(result.relevant_ranks) if result.relevant_ranks else None,
                "retrieved_ids": result.retrieved_ids[:10],
                "latency_s": round(latency, 3),
            }
        )
        log.info(
            "eval.question",
            id=q["id"],
            category=result.category,
            hit=result.hit,
            best_rank=min(result.relevant_ranks) if result.relevant_ranks else None,
        )
    return results, per_question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="curated", help="gold set name under eval/golden")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip Ragas; deterministic, free, needs no API key",
    )
    parser.add_argument("--k", type=int, default=10, help="cutoff for recall@k / ndcg@k")
    parser.add_argument("--out", type=Path, help="where to write the results JSON")
    args = parser.parse_args()

    configure_logging()
    gold_path = GOLDEN_DIR / f"{args.set}.jsonl"
    questions = load_questions(gold_path)

    if not args.retrieval_only and not settings.judge_configured:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Use --retrieval-only for the free, "
            "deterministic half of the benchmark."
        )

    started = time.monotonic()
    try:
        with EmbeddingsClient() as embedder:
            embedder.health()
            results, per_question = evaluate_retrieval(questions, embedder=embedder, k=args.k)
    finally:
        close_pool()

    payload: dict[str, Any] = {
        "provenance": provenance(retrieval_only=args.retrieval_only, gold_path=gold_path),
        "retrieval": aggregate(results, k=args.k),
        "per_question": per_question,
        "elapsed_s": round(time.monotonic() - started, 1),
    }

    if not args.retrieval_only:
        payload["generation"] = {
            "status": "not_implemented",
            "note": "Ragas metrics land with the judge-calibration work.",
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    r = payload["retrieval"]
    print()
    print(f"  n={r['n']} ({r['n_answerable']} answerable, {r['n_unanswerable']} unanswerable)")
    print(f"  recall@{args.k}  {r[f'recall@{args.k}']:.3f}")
    print(f"  ndcg@{args.k}    {r[f'ndcg@{args.k}']:.3f}")
    print(f"  mrr        {r['mrr']:.3f}")
    print(f"  hit rate   {r['hit_rate']:.3f}")
    if r["abstain_recall"] is not None:
        print(f"  abstain    {r['abstain_recall']:.3f}")
    for category, stats in r["by_category"].items():
        print(
            f"    {category:26s} n={stats['n']:<3d} recall@{args.k}={stats[f'recall@{args.k}']:.3f}"
        )
    print(f"\n  → {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

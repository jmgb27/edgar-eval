"""Decide whether an evaluation run passes, and say why in the job summary.

Exit 0 or 1, and write a markdown table to `--summary` so the GitHub Actions
run page itself becomes the evidence: every run shows the metrics it enforced
and what it compared them against. That costs a few lines of YAML and turns the
Actions tab into an artifact a reviewer can read without cloning anything.

A failing build also prints the worst individual questions with the chunk ids
that were retrieved, because a red gate that does not say which question broke
is a red gate people learn to re-run rather than read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="file to append a markdown summary to")
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    thresholds = yaml.safe_load(args.thresholds.read_text())
    current: dict[str, Any] = results["retrieval"]

    baseline: dict[str, Any] | None = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text()).get("retrieval")

    failures: list[str] = []
    rows: list[tuple[str, str, str, str, str, str]] = []

    floors: dict[str, float] = thresholds.get("absolute_floor", {})
    regressions: dict[str, float] = thresholds.get("max_regression", {})

    for metric, floor in floors.items():
        value = current.get(metric)
        if value is None:
            failures.append(f"`{metric}` missing from results")
            continue

        base = baseline.get(metric) if baseline else None
        delta = None if base is None else value - base
        status = "pass"

        if value < floor:
            failures.append(f"`{metric}` {value:.3f} is below the floor {floor:.3f}")
            status = "**FAIL — below floor**"

        tolerance = regressions.get(metric)
        if base is not None and tolerance is not None and (base - value) > tolerance:
            failures.append(
                f"`{metric}` regressed {base - value:.3f} from baseline {base:.3f} "
                f"(tolerance {tolerance:.3f})"
            )
            status = "**FAIL — regression**"

        rows.append(
            (
                metric,
                _fmt(base),
                _fmt(value),
                "—" if delta is None else f"{delta:+.3f}",
                _fmt(floor),
                status,
            )
        )

    # Per-category floors: tables are the hard case and must not be able to hide
    # inside a healthy average.
    by_category: dict[str, Any] = current.get("by_category", {})
    for category, limits in (thresholds.get("per_category_floor") or {}).items():
        stats = by_category.get(category)
        if stats is None:
            failures.append(f"category `{category}` missing from results")
            continue
        for metric, floor in limits.items():
            value = stats.get(metric)
            if value is None or value < floor:
                failures.append(
                    f"`{category}.{metric}` {_fmt(value)} is below the floor {floor:.3f}"
                )

    minimum = thresholds.get("min_samples")
    if minimum is not None and current.get("n", 0) < minimum:
        failures.append(f"only {current.get('n', 0)} questions scored; {minimum} required")

    # ── report ──────────────────────────────────────────────
    provenance = results.get("provenance", {})
    lines = [
        "## Retrieval evaluation",
        "",
        f"`{provenance.get('git_sha', '?')}` · gold set `{provenance.get('gold_set', '?')}` "
        f"(`{provenance.get('gold_set_sha256', '?')}`) · "
        f"embed `{provenance.get('embedding_model', '?')}` · "
        f"rerank `{provenance.get('reranker_model', '?')}` · "
        f"context header `{provenance.get('rerank_with_context')}`",
        "",
        f"n = {current.get('n')} ({current.get('n_answerable')} answerable, "
        f"{current.get('n_unanswerable')} unanswerable)",
        "",
        "| Metric | Baseline | Current | Delta | Floor | Status |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| `{m}` | {b} | {c} | {d} | {f} | {s} |" for m, b, c, d, f, s in rows]

    if by_category:
        lines += ["", "| Category | n | Recall@10 | nDCG@10 |", "|---|---|---|---|"]
        for category, stats in sorted(by_category.items()):
            lines.append(
                f"| {category} | {stats.get('n')} | "
                f"{_fmt(stats.get('recall@10'))} | {_fmt(stats.get('ndcg@10'))} |"
            )

    # Unanswerable questions have no gold span by construction, so listing
    # them here would report correct behaviour as a miss -- the exact kind of
    # misleading output that teaches a reviewer to distrust the whole report.
    worst = sorted(
        (
            q
            for q in results.get("per_question", [])
            if not q.get("hit") and q.get("category") != "unanswerable"
        ),
        key=lambda q: q.get("id", ""),
    )
    if worst:
        lines += ["", f"**{len(worst)} question(s) with no gold span retrieved:**", ""]
        lines += [
            f"- `{q['id']}` ({q['category']}) — {q['question']}  \n"
            f"  retrieved: `{q.get('retrieved_ids', [])[:6]}`"
            for q in worst[:8]
        ]

    if failures:
        lines += ["", "### Failures", ""] + [f"- {f}" for f in failures]
    else:
        lines += ["", "All thresholds met."]

    report = "\n".join(lines)
    print(report)
    if args.summary:
        with args.summary.open("a") as fh:
            fh.write(report + "\n")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

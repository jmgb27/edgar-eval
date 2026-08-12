"""Metadata filters, and the clamping that keeps them from emptying the corpus.

The single most common retrieval failure in a filing corpus is not a bad
embedding -- it is a filter for a year that was never ingested. A question
about "2024" against a corpus ending in 2023 produces an empty result set, and
an agent reading that empty set concludes the fact is absent from the filing
rather than that its own filter was wrong.

So extracted filters are clamped to what the corpus actually holds, and the
clamp is reported so the answer can say "the corpus covers FY2023" instead of
"I could not find it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edgar_eval.ingest.writer import corpus_bounds


@dataclass
class Filters:
    tickers: list[str] | None = None
    forms: list[str] | None = None
    items: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    notes: list[str] = field(default_factory=list)

    def as_params(self) -> dict[str, Any]:
        return {
            "tickers": self.tickers,
            "forms": self.forms,
            "items": self.items,
            "year_min": self.year_min,
            "year_max": self.year_max,
        }

    def is_empty(self) -> bool:
        return not any((self.tickers, self.forms, self.items, self.year_min, self.year_max))


def clamp_to_corpus(filters: Filters, bounds: dict[str, Any] | None = None) -> Filters:
    """Narrow filters to what exists, recording every adjustment in `notes`.

    Out-of-range years are clamped rather than dropped, so "2024 revenue"
    against a 2023 corpus retrieves FY2023 and the answer can say so. Unknown
    tickers are dropped entirely -- clamping a ticker would be a guess, and
    silently answering about the wrong company is far worse than saying the
    company is not in the corpus.
    """
    bounds = bounds if bounds is not None else corpus_bounds()
    if not bounds or not bounds.get("filings"):
        return filters

    out = Filters(
        tickers=list(filters.tickers) if filters.tickers else None,
        forms=list(filters.forms) if filters.forms else None,
        items=list(filters.items) if filters.items else None,
        year_min=filters.year_min,
        year_max=filters.year_max,
        notes=list(filters.notes),
    )

    corpus_min: int | None = bounds.get("year_min")
    corpus_max: int | None = bounds.get("year_max")
    # Both bounds come from the same aggregate, so they are either both present
    # or both NULL (an empty `filings` table). Checking them together keeps the
    # comparisons below total rather than relying on that invariant holding.
    if corpus_min is not None and corpus_max is not None:
        if out.year_min is not None and out.year_min > corpus_max:
            out.notes.append(
                f"requested fiscal year {out.year_min} is after the corpus "
                f"(FY{corpus_min}-FY{corpus_max}); clamped to FY{corpus_max}"
            )
            out.year_min, out.year_max = corpus_max, corpus_max
        elif out.year_max is not None and out.year_max < corpus_min:
            out.notes.append(
                f"requested fiscal year {out.year_max} is before the corpus "
                f"(FY{corpus_min}-FY{corpus_max}); clamped to FY{corpus_min}"
            )
            out.year_min, out.year_max = corpus_min, corpus_min

    known = {t.upper() for t in (bounds.get("tickers") or [])}
    if out.tickers and known:
        unknown = [t for t in out.tickers if t.upper() not in known]
        if unknown:
            kept = [t for t in out.tickers if t.upper() in known]
            out.notes.append(
                f"{', '.join(unknown)} not in the corpus (have: {', '.join(sorted(known))})"
            )
            # Dropping to None rather than to [] on purpose: an empty array
            # would match nothing, turning "I do not hold this company" into
            # "this fact does not exist".
            out.tickers = kept or None
    return out

"""Graph state.

`trace_notes` uses the same `Annotated[list, operator.add]` reducer idiom as
`ai-agents/boilerplate-agent/main.py`, so nodes append to it without stepping
on each other. It is rendered in the UI as a timeline, which is the surface
that shows a reviewer what the agent actually did rather than asking them to
trust the answer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Grade = Literal["sufficient", "partial", "insufficient"]
Scope = Literal["in_scope", "out_of_scope"]


class Citation(TypedDict):
    marker: str  # "S1"
    chunk_id: int
    ticker: str
    form_type: str
    fiscal_year: int
    item: str | None
    item_title: str | None
    source_url: str
    quote: str


class RagState(TypedDict, total=False):
    # input
    question: str
    session_id: str

    # planning
    subqueries: list[str]
    filters: dict[str, Any]
    filter_notes: list[str]
    scope: Scope

    # retrieval
    candidates: list[Any]  # Candidate, kept loose so the graph module stays importable
    contexts: list[Any]

    # control
    attempts: int
    regenerations: int
    grade: Grade
    grade_reason: str

    # output
    answer: str
    citations: list[Citation]
    groundedness: float
    unsupported_claims: list[str]
    extractive: bool

    # diagnostics — append-only
    trace_notes: Annotated[list[str], operator.add]

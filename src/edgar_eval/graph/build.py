"""Graph wiring.

Deliberately close to linear. `dispatchops-ai` already demonstrates a genuinely
agentic graph with human-in-the-loop interrupts and spend guardrails; repeating
that here would add branches to defend rather than retrieval quality to measure.
The interesting decisions in this project live in the SQL and the eval gate,
and the graph's job is to make them observable.

The one real loop -- grade, rewrite, retry -- exists because it fixes the most
common failure in a filing corpus: a first pass filtered too narrowly.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from edgar_eval.config import settings
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.graph import nodes
from edgar_eval.graph.state import RagState


def _route_after_analysis(state: RagState) -> Literal["retrieve", "abstain"]:
    return "abstain" if state.get("scope") == "out_of_scope" else "retrieve"


def _route_after_grade(state: RagState) -> Literal["generate", "rewrite", "abstain"]:
    grade = state.get("grade", "partial")
    attempts = state.get("attempts", 0)
    exhausted = attempts >= settings.max_retrieval_attempts

    if grade == "sufficient":
        return "generate"
    if not exhausted:
        return "rewrite"
    # Out of retries: a partial answer with citations beats an abstention,
    # but nothing at all beats a fabricated one.
    return "generate" if grade == "partial" and state.get("contexts") else "abstain"


def _route_after_verify(state: RagState) -> Literal["accept", "regenerate", "abstain"]:
    groundedness = state.get("groundedness", 1.0)
    if groundedness >= 0.7:
        return "accept"
    if groundedness < 0.4:
        return "abstain"
    return "regenerate" if state.get("regenerations", 0) == 0 else "accept"


def _regenerate(state: RagState) -> RagState:
    return {
        "regenerations": state.get("regenerations", 0) + 1,
        "trace_notes": ["regenerate: groundedness below threshold, retrying generation"],
    }


def build_graph(*, embedder: EmbeddingsClient, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(RagState)

    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("hybrid_retrieve", partial(nodes.hybrid_retrieve, embedder=embedder))
    graph.add_node("rerank", partial(nodes.rerank, embedder=embedder))
    graph.add_node("grade_context", nodes.grade_context)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify_groundedness", nodes.verify_groundedness)
    graph.add_node("mark_regeneration", _regenerate)
    graph.add_node("abstain", nodes.abstain)

    graph.set_entry_point("analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        _route_after_analysis,
        {"retrieve": "hybrid_retrieve", "abstain": "abstain"},
    )
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        _route_after_grade,
        {"generate": "generate", "rewrite": "rewrite_query", "abstain": "abstain"},
    )
    graph.add_edge("rewrite_query", "hybrid_retrieve")
    graph.add_edge("generate", "verify_groundedness")
    graph.add_conditional_edges(
        "verify_groundedness",
        _route_after_verify,
        {"accept": END, "regenerate": "mark_regeneration", "abstain": "abstain"},
    )
    graph.add_edge("mark_regeneration", "generate")
    graph.add_edge("abstain", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())

"""Embedding and reranking service.

Runs in its own container for two reasons:

  * Text Embeddings Inference (TEI) would be the obvious choice, but
    HuggingFace publishes no prebuilt CPU linux/arm64 image. On an Apple
    Silicon machine that means either a long Rust build from source or running
    the amd64 image under emulation -- ruinous for a compute-bound service.

  * Loading the models in-process inside the API container would mean the
    ingest CLI and the API each hold ~3.4 GB of weights, and the API's cold
    start would be dominated by a model load it does not need until the first
    query.

Weights are baked into the image at build time and `HF_HUB_OFFLINE=1` is set in
compose, so a change that reintroduced a runtime download would fail loudly
instead of quietly phoning home on someone else's laptop.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))

# bge-m3 is trained without an instruction prefix; bge-*-en-v1.5 wants one on
# the *query* side only. Keying the rule off the model name keeps a model swap
# honest -- switching EMBEDDING_MODEL without this table would silently degrade
# retrieval rather than fail.
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # Match the container's actual CPU allocation rather than trusting an env
    # default. Docker Desktop hands the VM a fraction of the host's cores (4 of
    # 10 on a stock M1 Max install), so a hardcoded thread count oversubscribes
    # and measurably *slows* inference through context switching. Explicit
    # EMBED_THREADS still wins if someone has tuned it.
    threads = int(os.getenv("EMBED_THREADS") or 0) or (os.cpu_count() or 1)
    torch.set_num_threads(threads)

    started = time.monotonic()
    _state["embedder"] = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    _state["embedder"].max_seq_length = MAX_LENGTH
    _state["reranker"] = CrossEncoder(RERANKER_MODEL, device="cpu", max_length=RERANK_MAX_LENGTH)
    # Renamed in sentence-transformers 5.7; the old name still works but emits
    # a FutureWarning. Prefer the new one, fall back so the image can be
    # rebuilt against an older pin without editing code.
    embedder = _state["embedder"]
    get_dim = getattr(embedder, "get_embedding_dimension", None) or (
        embedder.get_sentence_embedding_dimension
    )
    _state["dim"] = int(get_dim())
    _state["load_seconds"] = round(time.monotonic() - started, 1)
    _state["threads"] = threads
    yield
    _state.clear()


app = FastAPI(title="edgar-eval embeddings", version="0.1.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    # Only affects models that use an instruction prefix; see QUERY_PREFIXES.
    kind: Literal["query", "passage"] = "passage"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(min_length=1)
    top_k: int | None = None


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    if "embedder" not in _state:
        raise HTTPException(status_code=503, detail="models still loading")
    return {
        "status": "ok",
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "dim": _state["dim"],
        "load_seconds": _state["load_seconds"],
        "threads": _state["threads"],
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if "embedder" not in _state:
        raise HTTPException(status_code=503, detail="models still loading")

    texts = req.texts
    if req.kind == "query" and (prefix := QUERY_PREFIXES.get(EMBEDDING_MODEL)):
        texts = [prefix + t for t in texts]

    # normalize_embeddings=True so cosine distance in Postgres is a dot
    # product, and so the halfvec index sees unit vectors.
    vectors = _state["embedder"].encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return EmbedResponse(
        model=EMBEDDING_MODEL,
        dim=_state["dim"],
        embeddings=[[float(x) for x in row] for row in vectors],
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if "reranker" not in _state:
        raise HTTPException(status_code=503, detail="models still loading")

    pairs = [(req.query, doc) for doc in req.documents]
    # activation_fn default applies sigmoid for these models, giving 0..1
    # scores that are comparable across queries -- which is what makes a fixed
    # RERANK_MIN_SCORE threshold meaningful.
    scores = _state["reranker"].predict(pairs, batch_size=8, show_progress_bar=False)

    ranked = sorted(
        (RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)),
        key=lambda r: r.score,
        reverse=True,
    )
    if req.top_k is not None:
        ranked = ranked[: req.top_k]
    return RerankResponse(model=RERANKER_MODEL, results=ranked)

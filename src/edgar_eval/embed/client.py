"""HTTP client for the embeddings service.

Everything that needs a vector goes through here rather than importing
sentence-transformers, so the ingest CLI, the API and the eval harness all
share one model and one warm process.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from edgar_eval.config import settings

# Generous: a cold container spends ~40s loading weights, and an ingest batch
# of 256 long chunks is genuinely slow on CPU. Failing fast here would just
# turn a slow ingest into a broken one.
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)

_RETRY = retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)


@dataclass(frozen=True)
class RerankHit:
    index: int
    score: float


class EmbeddingsClient:
    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.embeddings_url).rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=_TIMEOUT)

    def __enter__(self) -> EmbeddingsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, object]:
        r = self._client.get("/healthz")
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    @_RETRY
    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]:
        """Embed `texts`.

        `kind` matters only for models that use an instruction prefix on the
        query side. Passing it wrong on such a model costs recall silently, so
        callers should always be explicit rather than relying on the default.
        """
        if not texts:
            return []
        r = self._client.post("/embed", json={"texts": texts, "kind": kind})
        r.raise_for_status()
        payload = r.json()

        if payload["dim"] != settings.embedding_dim:
            raise RuntimeError(
                f"embeddings service returned dim={payload['dim']} but the schema is "
                f"vector({settings.embedding_dim}). Re-run migrations after changing "
                "EMBEDDING_MODEL, and re-index the corpus."
            )
        return payload["embeddings"]  # type: ignore[no-any-return]

    def embed_one(self, text: str, *, kind: str = "passage") -> list[float]:
        return self.embed([text], kind=kind)[0]

    @_RETRY
    def rerank(
        self, query: str, documents: list[str], *, top_k: int | None = None
    ) -> list[RerankHit]:
        """Score `documents` against `query`, best first."""
        if not documents:
            return []
        r = self._client.post(
            "/rerank",
            json={"query": query, "documents": documents, "top_k": top_k},
        )
        r.raise_for_status()
        return [RerankHit(index=h["index"], score=h["score"]) for h in r.json()["results"]]

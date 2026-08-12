"""Every environment variable the system reads, in one place.

`load_dotenv()` runs at import time and *before* anything imports Langfuse.
Langfuse v4 reads its credentials from the environment on the first
`get_client()` call, so a `.env` loaded after that import is silently ignored
and tracing quietly does nothing. Importing this module first is the fix, and
it is why `observability.py` imports `settings` before it imports langfuse.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    Defaults are the ones that make `docker compose up` work with an
    unmodified `.env.example`, so a reviewer who changes nothing still gets a
    working retrieval stack.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Database ────────────────────────────────────────────
    database_url: str = "postgresql://edgar:edgar@localhost:5442/edgar"
    db_pool_min: int = 1
    db_pool_max: int = 8

    # ── Embeddings service ──────────────────────────────────
    embeddings_url: str = "http://localhost:8081"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_dim: int = 1024

    # ── SEC EDGAR ───────────────────────────────────────────
    edgar_user_agent: str = "edgar-eval you@example.com"
    edgar_rate_limit_per_sec: float = 5.0
    max_filing_bytes: int = 26_214_400

    # ── Generation model (optional) ─────────────────────────
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.7-plus"

    # ── Judge model (optional) ──────────────────────────────
    anthropic_api_key: str = ""
    judge_model: str = "claude-haiku-4-5"

    # ── Langfuse (optional) ─────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_tracing_environment: str = "development"

    # ── Retrieval tuning ────────────────────────────────────
    retrieval_pool: int = Field(default=40, description="per-arm candidates before fusion")
    retrieval_fused: int = Field(default=24, description="survivors of RRF, fed to reranker")
    retrieval_topk: int = Field(default=6, description="survivors of rerank, fed to generator")
    # Ablation switches. These exist so every row of the README table is
    # produced by the shipped pipeline under a different setting, rather than
    # by commenting code out and trusting the result.
    retrieval_mode: str = Field(default="hybrid", description="hybrid | dense | lexical")
    rerank_enabled: bool = True
    rerank_min_score: float = 0.05
    # Whether the cross-encoder sees the contextual header the embedder saw.
    # An ablation dimension, not a settled question -- see
    # docs/measurements.md 'Open question: rerank input'.
    rerank_with_context: bool = True
    hnsw_ef_search: int = 100
    max_retrieval_attempts: int = 2
    rrf_k: int = Field(default=60, description="Cormack et al. 2009 constant; do not tune casually")

    # ── Ingest ──────────────────────────────────────────────
    # Bumped when chunking or sectionising changes. Recorded on every filing so
    # a benchmark result can be attributed to the pipeline that produced it.
    ingest_version: str = "1"
    chunk_max_characters: int = 1800
    chunk_new_after_n_chars: int = 1400
    chunk_combine_under_n_chars: int = 400
    chunk_overlap: int = 200

    # ── Bootstrap ───────────────────────────────────────────
    seed_demo: bool = True

    @property
    def llm_configured(self) -> bool:
        """False means `/ask` runs in extractive mode instead of failing."""
        return bool(self.qwen_api_key)

    @property
    def judge_configured(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

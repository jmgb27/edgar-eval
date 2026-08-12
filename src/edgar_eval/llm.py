"""Generation model, and the extractive path used when there isn't one.

`/search` never touches this module -- hybrid retrieval and reranking are
entirely local. Only answer *synthesis* needs a model, so the absence of a key
degrades one endpoint rather than breaking the application. This mirrors the
`MOCK_AGENT` idiom in `dispatchops-ai/.env.example`.
"""

from __future__ import annotations

from typing import Any

from edgar_eval.config import settings
from edgar_eval.logging import get_logger

log = get_logger(__name__)


def llm_configured() -> bool:
    return settings.llm_configured


def build_llm(*, temperature: float = 0.0, max_tokens: int = 2048) -> Any:
    """Qwen through its OpenAI-compatible surface.

    Alibaba Cloud Model Studio speaks the OpenAI protocol, so `ChatOpenAI`
    talks to it directly -- only the base URL and model id change. The same
    code points at OpenRouter (`qwen/qwen3.7-plus`) or a self-hosted vLLM
    endpoint by editing .env.
    """
    if not settings.llm_configured:
        raise RuntimeError(
            "QWEN_API_KEY is not set. /search works without it; /ask falls back to "
            "extractive mode. Set it in .env for synthesised answers."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.qwen_model,
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=3,
    )


def build_judge(*, max_tokens: int = 2048) -> Any:
    """Claude Haiku for Ragas.

    Deliberately a different vendor from the generator: a judge from the same
    family as the model it grades shares its blind spots, and the resulting
    faithfulness score flatters itself.
    """
    if not settings.judge_configured:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; `make eval` needs it.")
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.judge_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=3,
    )

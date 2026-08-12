"""The optional-credential contract.

`/search` and `make eval-retrieval` are promised to work with no keys set.
That promise is only as good as the code paths that branch on these three
properties, so they get asserted rather than assumed.
"""

from __future__ import annotations

import pytest

from edgar_eval.config import Settings

# Credentials that must be absent for a test to be measuring the code rather
# than the developer's shell. `_env_file=None` alone is not enough:
# pydantic-settings still reads the live process environment, so a machine with
# ANTHROPIC_API_KEY exported would see `judge_configured is True` here and the
# no-credentials contract would silently stop being tested.
_CREDENTIAL_VARS = (
    "QWEN_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


def _settings(**overrides: str) -> Settings:
    with pytest.MonkeyPatch.context() as mp:
        for var in _CREDENTIAL_VARS:
            mp.delenv(var, raising=False)
        return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_no_credentials_is_a_valid_configuration() -> None:
    s = _settings()
    assert s.llm_configured is False
    assert s.judge_configured is False
    assert s.langfuse_configured is False


def test_llm_configured_tracks_qwen_key() -> None:
    assert _settings(qwen_api_key="sk-test").llm_configured is True


def test_judge_configured_tracks_anthropic_key() -> None:
    assert _settings(anthropic_api_key="sk-ant-test").judge_configured is True


def test_langfuse_needs_both_halves_of_the_keypair() -> None:
    """A half-configured Langfuse must read as disabled, not as enabled.

    Getting this backwards means the app tries to authenticate with a missing
    secret on every request and logs a failure per query.
    """
    assert _settings(langfuse_public_key="pk-only").langfuse_configured is False
    assert _settings(langfuse_secret_key="sk-only").langfuse_configured is False
    assert _settings(langfuse_public_key="pk", langfuse_secret_key="sk").langfuse_configured is True


def test_retrieval_stages_narrow_monotonically() -> None:
    """pool -> fused -> topk must shrink, or a stage is doing nothing."""
    s = _settings()
    assert s.retrieval_pool >= s.retrieval_fused > s.retrieval_topk

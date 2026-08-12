"""Shared fixtures.

Unit tests must not need Postgres, the embeddings service, or any API key --
that constraint is what lets the `unit` CI job run on forks. Anything that
needs a live service belongs in tests/integration and carries the
`integration` marker.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

_OPTIONAL_KEYS = (
    "QWEN_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run a test as if the reviewer had cloned the repo and set nothing.

    This is the configuration the README promises works, so it is the one the
    tests exercise by default rather than as an afterthought.
    """
    for key in _OPTIONAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Set env vars and clear the settings cache so they take effect."""
    from edgar_eval.config import get_settings

    def _set(**kwargs: str) -> None:
        for key, value in kwargs.items():
            monkeypatch.setenv(key.upper(), value)
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _set
    monkeypatch.undo()
    get_settings.cache_clear()


def pytest_configure(config: pytest.Config) -> None:
    # Keep a stray developer .env from leaking into unit-test expectations.
    os.environ.setdefault("EDGAR_EVAL_TESTING", "1")

"""Tracing must degrade to a no-op, and that is tested rather than asserted.

The README promises a stranger can clone this repo and run it with no
credentials. Langfuse is the easiest place for that promise to rot, because a
missing key there produces an exception deep inside a request rather than at
startup.
"""

from __future__ import annotations

import pytest

from edgar_eval import observability as obs


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from edgar_eval.config import get_settings

    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(obs, "settings", get_settings())
    yield
    get_settings.cache_clear()


def test_disabled_without_keys() -> None:
    assert obs.langfuse_enabled() is False


def test_run_callbacks_returns_an_empty_list() -> None:
    """Empty list, not None: it is passed straight into LangChain's
    `config={"callbacks": ...}`, so callers never branch on tracing."""
    assert obs.run_callbacks() == []


def test_trace_context_is_a_transparent_no_op() -> None:
    entered = False
    with obs.trace_context(trace_name="t", session_id="s"):
        entered = True
    assert entered


def test_score_is_silent() -> None:
    obs.score("groundedness-selfcheck", 0.9)  # must not raise


def test_current_trace_id_is_none() -> None:
    assert obs.current_trace_id() is None


def test_flush_is_silent() -> None:
    obs.flush()  # must not raise


def test_verify_auth_never_raises_and_reports_disabled() -> None:
    assert obs.verify_auth() is False


def test_partial_credentials_read_as_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Half a keypair must disable tracing, not attempt it on every request."""
    from edgar_eval.config import get_settings

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-only")
    get_settings.cache_clear()
    monkeypatch.setattr(obs, "settings", get_settings())
    try:
        assert obs.langfuse_enabled() is False
        assert obs.run_callbacks() == []
    finally:
        get_settings.cache_clear()

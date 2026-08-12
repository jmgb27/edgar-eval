"""Langfuse tracing — strictly optional.

A Python port of `dispatchops-ai/src/agent/observability.ts`, function for
function. The contract is the same: with no keys set, every function here is a
no-op and the application behaves identically. That is not politeness, it is
the difference between a stranger being able to run this repo and not.

Two ordering traps, both learned the hard way and both worth stating:

  * `load_dotenv()` must run *before* anything imports langfuse, because v4
    reads credentials from the environment on the first `get_client()` call.
    Importing `edgar_eval.config` at the top of this module is what guarantees
    it, so that import is load-bearing and must not be "tidied" away.

  * `auth_check()` warns rather than exits. The reference implementation in
    `boilerplate-agent/main.py` correctly hard-fails, because it is a demo
    script and a silently untraced demo is worthless. A *service* must not
    refuse to boot because an observability vendor is unreachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Imported first, and deliberately: it runs load_dotenv() at import time.
# Every langfuse import in this module is function-local so that ordering
# holds no matter how the import block is sorted.
from edgar_eval.config import settings
from edgar_eval.logging import get_logger

log = get_logger(__name__)

_auth_checked = False
_auth_ok = True


def langfuse_enabled() -> bool:
    """True only when both halves of the keypair are present and auth held."""
    return settings.langfuse_configured and _auth_ok


def verify_auth() -> bool:
    """Check credentials once at startup. Never raises.

    A failure disables tracing for the process rather than taking the service
    down with it.
    """
    global _auth_checked, _auth_ok
    if _auth_checked or not settings.langfuse_configured:
        return langfuse_enabled()
    _auth_checked = True
    try:
        from langfuse import get_client

        _auth_ok = bool(get_client().auth_check())
        if not _auth_ok:
            log.warning("langfuse.auth_failed", detail="tracing disabled for this process")
    except Exception as exc:
        _auth_ok = False
        log.warning("langfuse.unavailable", error=str(exc), detail="tracing disabled")
    return _auth_ok


def run_callbacks() -> list[Any]:
    """LangChain callbacks for a graph run. Empty list when tracing is off.

    An empty list is a valid `config={"callbacks": [...]}` value, so callers
    never branch on whether tracing is enabled.
    """
    if not langfuse_enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


@contextmanager
def trace_context(
    *,
    trace_name: str,
    session_id: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Group everything inside into one named, session-scoped trace."""
    if not langfuse_enabled():
        yield
        return
    from langfuse import propagate_attributes

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        tags=tags or [],
        metadata=metadata or {},
    ):
        yield


def score(name: str, value: float, *, comment: str | None = None) -> None:
    """Attach a score to the current trace. Silent when tracing is off."""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:
        # Losing a score must never fail a query.
        log.debug("langfuse.score_failed", name=name, error=str(exc))


def current_trace_id() -> str | None:
    """The active trace id, for returning to the browser so a thumbs-up can be
    posted back against it."""
    if not langfuse_enabled():
        return None
    try:
        from langfuse import get_client

        trace_id = get_client().get_current_trace_id()
        return str(trace_id) if trace_id else None
    except Exception:
        return None


def flush() -> None:
    """Force-send buffered events. Call on shutdown and at the end of scripts."""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        log.debug("langfuse.flush_failed", error=str(exc))

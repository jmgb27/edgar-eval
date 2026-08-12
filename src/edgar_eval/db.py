"""Postgres connection pool.

A single module-level pool, opened lazily and closed by the FastAPI lifespan
hook (and by `close_pool()` at the end of every CLI script). psycopg3's
`ConnectionPool` is threadsafe, so the sync API is fine here -- the expensive
work in this system is the embedding service and the model call, neither of
which holds a connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from edgar_eval.config import settings

_pool: ConnectionPool[Connection[dict[str, Any]]] | None = None


def _configure(conn: Connection[dict[str, Any]]) -> None:
    """Teach this connection about the `vector` type.

    Without this, psycopg round-trips embeddings as strings and every insert
    pays a parse on the server side.

    The `except` is load-bearing rather than defensive: `register_vector` looks
    the type up in `pg_type`, so it fails on a database where the extension has
    not been created yet -- and the connection that runs migration 001, which
    creates it, is itself one of these connections. Without the fallback the
    pool cannot hand out the connection needed to bootstrap the database it is
    connected to. Every connection opened after 001 registers normally.
    """
    try:
        register_vector(conn)
    except Exception:
        conn.rollback()


def get_pool() -> ConnectionPool[Connection[dict[str, Any]]]:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator[Connection[dict[str, Any]]]:
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

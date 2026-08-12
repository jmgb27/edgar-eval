"""Migrations are an artifact reviewers read. Keep them well-formed."""

from __future__ import annotations

import re

import pytest

from edgar_eval.config import REPO_ROOT

MIGRATIONS = sorted((REPO_ROOT / "db" / "migrations").glob("*.sql"))
NAME_RE = re.compile(r"^(\d{3})_[a-z0-9_]+$")


def test_migrations_exist() -> None:
    assert MIGRATIONS, "no migrations found"


@pytest.mark.parametrize("path", MIGRATIONS, ids=lambda p: p.stem)
def test_migration_is_well_named(path) -> None:  # type: ignore[no-untyped-def]
    assert NAME_RE.match(path.stem), (
        f"{path.name} must be NNN_lower_snake.sql so ordering is lexicographic"
    )


def test_migration_numbers_are_unique_and_contiguous() -> None:
    numbers = [int(p.stem[:3]) for p in MIGRATIONS]
    assert len(set(numbers)) == len(numbers), "duplicate migration number"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"migration numbers must start at 001 and not skip: {numbers}"
    )


def test_extension_migration_runs_first() -> None:
    """CREATE EXTENSION vector has to precede any table using the type."""
    assert "CREATE EXTENSION" in MIGRATIONS[0].read_text()


def test_chunks_embedding_dimension_matches_settings() -> None:
    """The DDL and the configured embedding dimension must not drift apart.

    Swapping to a model with a different width without editing the schema
    would otherwise fail at first insert, deep inside an ingest run, rather
    than here.
    """
    from edgar_eval.config import settings

    ddl = next(p for p in MIGRATIONS if p.stem.endswith("chunks")).read_text()
    assert f"vector({settings.embedding_dim})" in ddl

    indexes = next(p for p in MIGRATIONS if p.stem.endswith("indexes")).read_text()
    assert f"halfvec({settings.embedding_dim})" in indexes

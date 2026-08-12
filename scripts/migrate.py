"""Apply numbered SQL migrations.

There is no Alembic here on purpose. Alembic earns its keep by autogenerating
diffs from SQLAlchemy models; this schema has no ORM models, and what it does
have -- an expression index over a halfvec cast, a generated weighted tsvector
column, non-default HNSW storage parameters -- is exactly what autogenerate
handles badly. Sixty lines of runner keeps the DDL readable as the artifact it
is.

Applied migrations are recorded with the sha256 of their contents. Editing a
migration that has already run is an error rather than a silent no-op, because
the alternative is two databases with the same version number and different
schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from edgar_eval.config import REPO_ROOT, settings
from edgar_eval.db import close_pool, connection
from edgar_eval.logging import configure_logging, get_logger

log = get_logger(__name__)

MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    sha256     char(64)    NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def _discover() -> list[Path]:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise SystemExit(f"no migrations found in {MIGRATIONS_DIR}")
    return files


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(*, dry_run: bool = False) -> int:
    applied_count = 0
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
            cur.execute("SELECT version, sha256 FROM schema_migrations")
            applied = {row["version"]: row["sha256"] for row in cur.fetchall()}
        conn.commit()

        for path in _discover():
            version = path.stem
            digest = _sha256(path)

            if version in applied:
                if applied[version] != digest:
                    raise SystemExit(
                        f"migration {version} has already been applied but its contents "
                        f"changed (recorded {applied[version][:12]}, on disk {digest[:12]}). "
                        "Add a new migration instead of editing an applied one."
                    )
                log.debug("migration.skip", version=version)
                continue

            if dry_run:
                log.info("migration.pending", version=version)
                applied_count += 1
                continue

            log.info("migration.apply", version=version)
            with conn.cursor() as cur:
                cur.execute(path.read_text())
                cur.execute(
                    "INSERT INTO schema_migrations (version, sha256) VALUES (%s, %s)",
                    (version, digest),
                )
            conn.commit()
            applied_count += 1

    return applied_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply database migrations.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list migrations that would run, without running them",
    )
    args = parser.parse_args()

    configure_logging()
    log.info("migrate.start", database=settings.database_url.rsplit("@", 1)[-1])

    try:
        n = migrate(dry_run=args.dry_run)
    finally:
        close_pool()

    verb = "pending" if args.dry_run else "applied"
    log.info("migrate.done", **{verb: n})
    return 0


if __name__ == "__main__":
    sys.exit(main())

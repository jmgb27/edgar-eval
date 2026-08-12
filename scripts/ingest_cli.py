"""Ingest SEC filings into the corpus.

uv run python scripts/ingest_cli.py --ticker AAPL --form 10-K --years 2023 2023
uv run python scripts/ingest_cli.py --ticker AAPL MSFT --form 10-K --years 2023 2023
"""

from __future__ import annotations

import argparse
import sys
import time

from edgar_eval.db import close_pool
from edgar_eval.embed.client import EmbeddingsClient
from edgar_eval.ingest.edgar import EdgarClient
from edgar_eval.ingest.pipeline import ingest_filing
from edgar_eval.ingest.writer import corpus_bounds
from edgar_eval.logging import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", nargs="+", required=True, help="one or more tickers")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"])
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("FROM", "TO"),
        help="inclusive fiscal-year range (not filing-date year)",
    )
    parser.add_argument("--limit", type=int, help="max filings per ticker")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="do not delete a filing's existing chunks before writing new ones",
    )
    args = parser.parse_args()

    configure_logging()
    years = (args.years[0], args.years[1]) if args.years else None
    started = time.monotonic()
    total_chunks = 0

    try:
        with EdgarClient() as edgar, EmbeddingsClient() as embedder:
            embedder.health()  # fail early and clearly, not mid-filing
            for ticker in args.ticker:
                filings = edgar.find_filings(
                    ticker, form_type=args.form, years=years, limit=args.limit
                )
                if not filings:
                    log.warning("ingest.no_filings", ticker=ticker, form=args.form, years=years)
                    continue
                for filing in filings:
                    result = ingest_filing(
                        filing,
                        edgar=edgar,
                        embedder=embedder,
                        replace=not args.keep_existing,
                    )
                    total_chunks += result.written
    finally:
        close_pool()

    elapsed = time.monotonic() - started
    log.info("ingest.complete", chunks=total_chunks, seconds=round(elapsed, 1))
    print()
    print(f"  {total_chunks} chunks in {elapsed:.0f}s")
    print(f"  corpus: {corpus_bounds()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

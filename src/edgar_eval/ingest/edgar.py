"""SEC EDGAR client.

Access policy, because it is a condition of use rather than a nicety:

  * Every request carries a `User-Agent` naming the tool and a real contact
    address. SEC blocks requests without one.
  * Requests are throttled to EDGAR_RATE_LIMIT_PER_SEC (default 5/s, half the
    published 10/s ceiling), enforced by a token bucket shared across the
    process rather than by a sleep between calls.
  * Responses are cached on disk, so re-running an ingest costs SEC nothing.

Only the *primary document* of a filing is fetched -- the single HTML file that
is the 10-K itself -- never the R-files or FilingSummary. See
`docs/data-and-terms.md`.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from edgar_eval.config import REPO_ROOT, settings
from edgar_eval.logging import get_logger

log = get_logger(__name__)

FormType = Literal["10-K", "10-Q"]

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

CACHE_DIR = REPO_ROOT / "data" / "cache"


class RateLimiter:
    """Token bucket, shared across threads.

    A plain `sleep(1/rate)` between calls throttles a single sequential caller
    but does nothing once two threads are fetching, which is exactly when
    exceeding SEC's limit would get the contact address blocked.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self._min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


@dataclass(frozen=True)
class Filing:
    """One filing, resolved far enough to be fetched."""

    accession_no: str
    cik: str
    ticker: str
    company_name: str
    form_type: FormType
    filing_date: date
    period_end_date: date
    primary_document: str
    fiscal_year: int
    fiscal_period: str

    @property
    def source_url(self) -> str:
        return ARCHIVE_URL.format(
            cik=int(self.cik),
            accession_nodash=self.accession_no.replace("-", ""),
            document=self.primary_document,
        )


def fiscal_period_for(form_type: FormType, period_end: date, fiscal_year_end_month: int) -> str:
    """Label the period as FY or Q1..Q4.

    A 10-K is always FY. For a 10-Q the quarter is counted from the company's
    own fiscal year end, not from the calendar -- Apple's fiscal year ends in
    September, so its December quarter is Q1, not Q4. Getting this wrong makes
    "the most recent quarter" retrieve the wrong filing.
    """
    if form_type == "10-K":
        return "FY"
    months_since_fy_start = (period_end.month - fiscal_year_end_month - 1) % 12
    return f"Q{months_since_fy_start // 3 + 1}"


def fiscal_year_for(form_type: FormType, period_end: date, fiscal_year_end_month: int) -> int:
    """The fiscal year a period belongs to.

    A period ending after the fiscal year-end month belongs to the *next*
    fiscal year. Apple's quarter ending 2023-12-30 is fiscal 2024.
    """
    if period_end.month > fiscal_year_end_month:
        return period_end.year + 1
    return period_end.year


class EdgarClient:
    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(settings.edgar_rate_limit_per_sec)
        self._client = httpx.Client(
            headers={
                # SEC requires a descriptive UA with a contact address and
                # blocks requests without one.
                "User-Agent": settings.edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            follow_redirects=True,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ── transport ───────────────────────────────────────────
    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> bytes:
        self._limiter.acquire()
        response = self._client.get(url)
        if response.status_code == 403:
            raise RuntimeError(
                "SEC returned 403. This almost always means EDGAR_USER_AGENT is missing or "
                "does not contain a real contact address. Current value: "
                f"{settings.edgar_user_agent!r}"
            )
        response.raise_for_status()
        return response.content

    def _cached(self, key: str, url: str) -> bytes:
        path = self._cache_dir / key
        if path.exists():
            return path.read_bytes()
        payload = self._get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return payload

    # ── lookups ─────────────────────────────────────────────
    def cik_for_ticker(self, ticker: str) -> str:
        raw = self._cached("company_tickers.json", TICKERS_URL)
        table: dict[str, dict[str, Any]] = json.loads(raw)
        wanted = ticker.upper()
        for entry in table.values():
            if str(entry["ticker"]).upper() == wanted:
                return f"{int(entry['cik_str']):010d}"
        raise LookupError(f"ticker {ticker!r} not found in SEC's company_tickers.json")

    def submissions(self, cik: str) -> dict[str, Any]:
        raw = self._cached(f"submissions/CIK{cik}.json", SUBMISSIONS_URL.format(cik=int(cik)))
        return json.loads(raw)  # type: ignore[no-any-return]

    def find_filings(
        self,
        ticker: str,
        *,
        form_type: FormType = "10-K",
        years: tuple[int, int] | None = None,
        limit: int | None = None,
    ) -> list[Filing]:
        """Resolve a ticker to filings, newest first.

        `years` filters on *fiscal* year, which is what a question means by
        "FY2023" -- not on the filing date, which for a 10-K falls in the
        following calendar year.
        """
        cik = self.cik_for_ticker(ticker)
        data = self.submissions(cik)
        company_name = data["name"]

        # "--09-30" -> 9. Companies with a calendar year end report "--12-31".
        fye = data.get("fiscalYearEnd") or "1231"
        fye_month = int(str(fye).replace("-", "")[:2] or 12)

        recent = data["filings"]["recent"]
        found: list[Filing] = []
        for i, form in enumerate(recent["form"]):
            if form != form_type:
                continue
            report_date = recent["reportDate"][i]
            if not report_date:
                continue
            period_end = date.fromisoformat(report_date)
            fiscal_year = fiscal_year_for(form_type, period_end, fye_month)
            if years and not (years[0] <= fiscal_year <= years[1]):
                continue

            found.append(
                Filing(
                    accession_no=recent["accessionNumber"][i],
                    cik=cik,
                    ticker=ticker.upper(),
                    company_name=company_name,
                    form_type=form_type,
                    filing_date=date.fromisoformat(recent["filingDate"][i]),
                    period_end_date=period_end,
                    primary_document=recent["primaryDocument"][i],
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period_for(form_type, period_end, fye_month),
                )
            )
            if limit and len(found) >= limit:
                break
        return found

    # ── documents ───────────────────────────────────────────
    def fetch_document(self, filing: Filing) -> str:
        raw = self._cached(
            f"filings/{filing.accession_no}/{filing.primary_document}", filing.source_url
        )
        if len(raw) > settings.max_filing_bytes:
            log.warning(
                "edgar.filing_large",
                accession=filing.accession_no,
                bytes=len(raw),
                limit=settings.max_filing_bytes,
            )
        return raw.decode("utf-8", errors="replace")


_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HIDDEN_RE = re.compile(
    r"<(div|span)\b[^>]*style=\"[^\"]*display:\s*none[^\"]*\"[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_IX_HEADER_RE = re.compile(r"<ix:header\b.*?</ix:header>", re.IGNORECASE | re.DOTALL)


def strip_noise(html: str) -> str:
    """Remove the parts of an inline-XBRL filing that carry no readable text.

    Modern 10-Ks are inline XBRL: a hidden `<ix:header>` block holds thousands
    of machine-readable facts, and hidden spans carry tagged values duplicated
    from the visible text. Left in, they dominate the element stream, blow up
    memory, and pollute chunks with markup that reads as gibberish.
    """
    html = _IX_HEADER_RE.sub(" ", html)
    html = _SCRIPT_RE.sub(" ", html)
    html = _HIDDEN_RE.sub(" ", html)
    return html

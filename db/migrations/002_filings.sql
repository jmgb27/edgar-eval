-- One row per ingested SEC filing.
--
-- `accession_no` is the SEC's own primary key for a filing (e.g.
-- 0000320193-23-000106) and is stable forever, which makes it the right
-- natural key: re-ingesting the same filing is idempotent without a surrogate.
CREATE TABLE IF NOT EXISTS filings (
    accession_no    text PRIMARY KEY,
    cik             text        NOT NULL,
    ticker          text        NOT NULL,
    company_name    text        NOT NULL,
    form_type       text        NOT NULL CHECK (form_type IN ('10-K', '10-Q')),
    fiscal_year     smallint    NOT NULL,
    fiscal_period   text        NOT NULL CHECK (fiscal_period IN ('FY','Q1','Q2','Q3','Q4')),
    filing_date     date        NOT NULL,
    period_end_date date        NOT NULL,
    source_url      text        NOT NULL,

    -- Bumped whenever the chunking or sectionising logic changes, so a
    -- re-ingest can be told apart from a stale one in the eval provenance.
    ingest_version  text        NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS filings_ticker_year ON filings (ticker, fiscal_year DESC);

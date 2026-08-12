-- One row per retrievable chunk.
--
-- Two deliberate denormalisations, both for the hot path:
--
--   1. ticker / form_type / fiscal_year / fiscal_period / period_end_date are
--      copied down from `filings` so a filtered hybrid search never joins.
--      The filter runs before the vector scan, and a join there costs more
--      than the duplication.
--
--   2. `text` and `embed_text` are stored separately. `text` is what a human
--      and the generator see. `embed_text` is what was actually embedded --
--      the same text with a contextual header prepended:
--
--          Apple Inc. (AAPL) - 10-K - FY2023 - Item 7 - MD&A
--          Section: Results of Operations > Segment Operating Performance
--
--      Without that header a chunk reading "increased 8% year over year" is
--      unretrievable: it names neither the company, the year, nor the metric.
--      Storing both means we can re-embed without re-chunking, and the header
--      never leaks into the answer.
CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    accession_no    text     NOT NULL REFERENCES filings(accession_no) ON DELETE CASCADE,

    -- denormalised filter columns (see note 1 above)
    ticker          text     NOT NULL,
    form_type       text     NOT NULL,
    fiscal_year     smallint NOT NULL,
    fiscal_period   text     NOT NULL,
    period_end_date date     NOT NULL,

    -- structure
    item            text,                             -- '1A', '7', '7A', 'COVER'
    item_title      text,                             -- canonical, not the filing's own wording
    section_path    text[]   NOT NULL DEFAULT '{}',   -- breadcrumb of enclosing Title elements

    -- content (see note 2 above)
    text            text     NOT NULL,
    embed_text      text     NOT NULL,
    table_html      text,                             -- NULL unless has_table
    has_table       boolean  NOT NULL DEFAULT false,
    element_types   text[]   NOT NULL DEFAULT '{}',

    -- provenance
    chunk_index     int      NOT NULL,
    element_index   int      NOT NULL,
    page_number     int,                              -- NULL for HTML sources
    source_url      text     NOT NULL,                -- includes #anchor when available
    token_count     int      NOT NULL,

    content_sha256  char(64) NOT NULL,
    embedding       vector(1024) NOT NULL,

    -- Weighted lexical vector. The item title carries weight A so a query
    -- naming a section ("risk factors") ranks that section's chunks above
    -- passing mentions of the phrase elsewhere in the filing.
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(item_title, '')), 'A') ||
        setweight(to_tsvector('english', text), 'B')
    ) STORED,

    created_at timestamptz NOT NULL DEFAULT now(),

    -- Makes re-ingesting a filing idempotent: identical content in the same
    -- filing collapses to one row instead of duplicating the corpus.
    CONSTRAINT chunks_content_uniq UNIQUE (accession_no, content_sha256)
);

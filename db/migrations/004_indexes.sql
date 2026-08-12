-- Vector index.
--
-- The column stores a full-precision vector(1024); the index is built over a
-- halfvec cast. That halves index size and build time. Recall loss at 1024
-- dimensions is under half a percent because cosine similarity over
-- normalised embeddings is nowhere near float16's precision floor -- and the
-- stored vector stays exact, so any future re-index (a different m, a
-- different operator class) starts from full precision rather than from
-- already-truncated data.
--
-- m=16 is the default. ef_construction=200 is not (the default is 64): at
-- corpus sizes in the tens of thousands the build takes seconds either way,
-- so the higher value is free recall.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Lexical index for the other half of hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Metadata filters. Composite ordering matches the query's WHERE clause:
-- ticker is nearly always bound, fiscal_year often, form_type sometimes.
CREATE INDEX IF NOT EXISTS chunks_ticker_year
    ON chunks (ticker, fiscal_year DESC, form_type);

-- Partial: roughly nothing queries for chunks with no Item, so the NULLs are
-- dead weight in the index.
CREATE INDEX IF NOT EXISTS chunks_item ON chunks (item) WHERE item IS NOT NULL;

CREATE INDEX IF NOT EXISTS chunks_period_end ON chunks (period_end_date DESC);
CREATE INDEX IF NOT EXISTS chunks_accession  ON chunks (accession_no);

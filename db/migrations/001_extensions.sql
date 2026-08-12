-- pgvector 0.8+ is required: the `hnsw.iterative_scan` GUC used by the hybrid
-- query does not exist before 0.8, and without it a filtered vector search
-- silently returns far fewer rows than requested whenever the filter is
-- selective (all the nearest neighbours belong to the wrong company).
CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram index support for the company/ticker fuzzy lookup in the UI.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

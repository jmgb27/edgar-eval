-- Thumbs from the UI. Stored locally as well as forwarded to Langfuse as a
-- score, so the signal survives Langfuse being disabled -- which is the
-- default configuration for anyone who just cloned the repo.
CREATE TABLE IF NOT EXISTS feedback (
    id           bigserial PRIMARY KEY,
    trace_id     text,                  -- NULL when tracing is disabled
    session_id   text        NOT NULL,
    question     text        NOT NULL,
    answer       text        NOT NULL,
    rating       smallint    NOT NULL CHECK (rating IN (-1, 1)),
    comment      text,
    chunk_ids    bigint[]    NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_session ON feedback (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS feedback_rating  ON feedback (rating, created_at DESC);

# edgar-eval

> RAG over SEC filings that publishes its own accuracy — and a CI gate that fails the build when retrieval gets worse.

**Status: under construction.** This README is a stub. It gets written last, in Phase 7, once every number in it is real and reproducible.

The benchmark table below is deliberately empty. Each blank cell is a piece of work that has not been done yet, and it stays blank until `make eval` produces the number. Filling a cell by hand would defeat the entire point of the project.

| Configuration | Recall@10 | Faithfulness | Answer correctness | Refuses unanswerable | p50 latency / $ per 100q |
|---|---|---|---|---|---|
| Baseline — fixed 512-token chunks, dense-only | | | | | |
| \+ structure-aware chunking (tables intact) | | | | | |
| \+ hybrid retrieval (dense + Postgres FTS, RRF) | | | | | |
| **\+ BGE reranker (top-24 → top-6)** | | | | | |
| Full pipeline **minus** reranker | | | | | |

## Build progress

- [x] **Phase 0** — skeleton: uv project, ruff/mypy/pytest, compose, schema, migrations
- [x] **Phase 1** — embeddings service (BGE, weights baked into the image)
- [x] **Phase 2** — ingestion: EDGAR → partition → sectionise → chunk
- [x] **Phase 3** — hybrid retrieval: RRF over dense + lexical, then reranking
- [x] **Phase 4** — LangGraph, FastAPI, single-file UI
- [x] **Phase 5** — Langfuse-optional observability
- [ ] **Phase 6** — evaluation: curated gold set, Ragas, judge calibration
- [ ] **Phase 7** — CI eval gate, and the README you are supposed to read

## Local development

```bash
make install     # uv sync
make up          # docker compose up -d  (postgres)
make migrate     # apply db/migrations/*.sql
make check       # ruff + mypy --strict + pytest
```

Requires Python 3.12 and Docker. Python 3.12 rather than 3.10 because
`unstructured` 0.25 requires ≥3.11, and 3.10 reaches end of life in October 2026.

## Licence

MIT — see [LICENSE](LICENSE).

Filing data is fetched from [SEC EDGAR](https://www.sec.gov/edgar) at run time and is
not redistributed in this repository; only accession numbers are committed. SEC
filings are US government works in the public domain. See `docs/data-and-terms.md`
(Phase 7) for the access policy this repo follows.

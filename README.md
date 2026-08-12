# edgar-eval

**RAG over SEC filings that publishes its own accuracy — and a CI gate that fails the build when retrieval gets worse.**

[![CI](https://github.com/jmgb27/edgar-eval/actions/workflows/ci.yaml/badge.svg)](https://github.com/jmgb27/edgar-eval/actions/workflows/ci.yaml)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

| Configuration | Recall@10 | nDCG@10 | MRR | Citation (filing / section) |
|---|---|---|---|---|
| **Dense only, no reranker** | **0.940** | **0.734** | **0.661** | 1.000 / **0.840** |
| \+ hybrid retrieval (dense + Postgres FTS, RRF) | 0.860 | 0.507 | 0.393 | 1.000 / 0.760 |
| \+ BGE reranker — **the shipped pipeline** | 0.920 | 0.654 | 0.579 | 1.000 / 0.680 |
| Dense + reranker (hybrid removed) | 0.920 | 0.654 | 0.579 | 1.000 / 0.680 |

> n=25 answerable questions (30 total, 5 unanswerable) over 2 filings — accession numbers in
> [`eval/golden/curated.jsonl`](eval/golden/curated.jsonl) · retrieval metrics are deterministic
> code over gold spans, **no API key, no LLM judge** · faithfulness judge separately calibrated at
> **TPR 1.000 / TNR 1.000** ([report](eval/judge_calibration_report.md)) · regenerate with
> `make eval-retrieval` · [raw results](eval/results/latest.json)

**The simplest configuration wins on every metric, and the pipeline this project was built to
demonstrate loses.** That result is at the top rather than buried, because a benchmark that only
ever confirmed its own design would not be worth running. What it means, and what it does not, is
in [The result that does not flatter the design](#the-result-that-does-not-flatter-the-design).

Built with Unstructured · BGE (bge-m3 + bge-reranker-base) · pgvector 0.8 · LangChain 1.x ·
LangGraph · Langfuse · Claude Haiku 4.5 as judge.

---

## In plain English

- A **10-K** is the annual report a US public company must file with the SEC. Apple's is about
  120 pages of prose and financial tables.
- Asking questions of one means **retrieval**: find the handful of paragraphs that answer the
  question, then answer only from those.
- Naive systems get financial tables wrong, because chunking text by length splits a table
  mid-row, and half a table is worse than none.
- Everything here is **measured**. Nothing in that table was typed by hand; `make eval-retrieval`
  regenerates it, and CI fails the build if the numbers get worse.
- **You do not need an API key** to reproduce the retrieval half. Embedding and reranking run
  locally on CPU.

## The failure this fixes

Apple's effective tax rate lives in a table that survives chunking as **157 characters**:

```
2023 2022 2021 Provision for income taxes $ 16,741 $ 19,300 $ 14,527 Effective tax rate 14.7 %
```

Ask *"What was the effective tax rate?"* and the reranker originally returned the **cash-flow
statement** — a passage containing no tax content at all — ranked first at 0.801. It won by being a
large grid of numbers.

The cause was an inconsistency: chunks are **embedded** with a contextual header naming the
company, fiscal year and Item, but the reranker was scoring the **bare** text. Stripped of its
header, the tax table is an anonymous grid. Fixing it removed the wrong winner. Measured across
25 questions, the header is worth **+0.180 recall and +0.284 nDCG** and wins in every category —
though judged on that one query alone it looked worse, which is the whole argument for having an
eval set. See [`docs/measurements.md`](docs/measurements.md).

## Quick start

```bash
git clone https://github.com/jmgb27/edgar-eval && cd edgar-eval
cp .env.example .env          # works unmodified — no keys required
make up                       # postgres + embeddings  (first build ~10 min, then ~60s)
make migrate
make corpus                   # ingest AAPL + MSFT FY2023 10-Ks from EDGAR
make eval-retrieval           # reproduces the Recall / nDCG / MRR / citation columns
```

**No API key is needed for any of that.** BGE embedding and reranking run on CPU in a container
whose weights are baked into the image, so ingestion, retrieval, the `/search` endpoint and the
retrieval half of the benchmark all work offline with the shipped `.env`.

| Needs nothing | Needs a key |
|---|---|
| `make eval-retrieval`, `make gate`, `/search`, full ingestion | Synthesised answers (`QWEN_API_KEY`), judge-scored metrics (`ANTHROPIC_API_KEY`) |

Without `QWEN_API_KEY`, `/ask` degrades to **extractive mode**: it returns the top-ranked source
passage verbatim behind a banner, rather than failing.

Requires Docker (allocate ≥4 CPU / 8 GB) and Python 3.12. First build is ~10 minutes because
~3.4 GB of model weights are baked into the image; subsequent starts are under a minute.

## How it works

```
EDGAR ──► partition_html ──► sectionise by Item ──► chunk per section ──► embed ──► pgvector
                                                                                      │
question ──► analyse ──► hybrid retrieve ──► rerank ──► grade ──► generate ──► verify ─┘
                         (dense ∪ lexical,   (BGE cross-   │
                          fused by RRF)       encoder)     └── rewrite & retry (max 2)
```

Five decisions that are choices rather than defaults:

- **Sectionise before chunking.** `chunk_by_title` runs per Item, so no chunk spans the Item
  1A/1B boundary. A chunk that straddles two Items cannot be filtered to either.
- **A contextual header is embedded but not stored** in the visible text. *"increased 8% year over
  year"* names neither company nor metric and is otherwise unretrievable.
- **RRF, not weighted score fusion.** Postgres `ts_rank_cd` has no IDF and no length normalisation
   — it is **not BM25**, and calling it that would be wrong. RRF reads only rank, so the lexical
  arm's uncalibrated scores cannot skew the result.
- **The lexical arm ORs its terms.** Under `websearch_to_tsquery`'s default AND, *"net cash
  provided by operating activities"* matched **zero** chunks, because Apple writes *"generated"*.
  One synonym silently reduced hybrid retrieval to dense-only.
- **`hnsw.iterative_scan = relaxed_order`** (pgvector 0.8+). Without it a selective metadata
  filter returns a handful of rows, because the vector scan picks candidates before the WHERE
  clause applies.

## The result that does not flatter the design

Plain dense retrieval beats the full pipeline on every metric in the table above.

**The aggregate gap is noise.** One question either way at n=25 moves recall by 0.04. Dense-only
finds `rsk-01`; the full pipeline finds `tmp-03`. Nothing there is readable.

**The ranking difference is not noise**, because it is systematic by category:

| | rank change, dense-only → full pipeline |
|---|---|
| Promoted (narrative, comparison) | `cmp-02` 2→1 · `cmp-03` 2→1 · `rsk-02` 3→1 · `sf-03` 4→2 |
| Demoted (tables, temporal) | `tbl-02` 1→4 · `tbl-03` 1→4 · `tbl-07` 2→7 · `tmp-02` 2→7 |

The reranker helps prose and hurts tables, consistently — the same weakness citation-section
accuracy reports independently (0.840 → 0.680). The likely mechanism: a 157-character table plus a
120-character contextual header is *mostly header*, so the cross-encoder sees least distinguishing
content for exactly the chunks that matter most.

**The default is deliberately unchanged.** One corpus, n=25, and a span-containment metric that
rewards finding text rather than answering well. Switching on that evidence would be the same
anecdote-driven tuning the contextual-header ablation already caught, pointing the other way.
[Next steps are written down](docs/measurements.md) — a shortened rerank header, a
minimum-matching-terms threshold instead of blanket OR, and a larger eval set.

## Evaluation

**The gold set is 30 hand-checkable questions** across six categories — `single_fact`,
`table_lookup`, `multi_filing_comparison`, `temporal`, `risk_factor_synthesis` and 5
`unanswerable`. Every question was drafted from facts extracted from the ingested filings, and
every gold span is **machine-verified to exist verbatim in the corpus** by
`make validate-goldset`, which runs before every evaluation and in CI. It caught an invented span
on its first run.

**Retrieval metrics are judge-free.** Recall@k, nDCG@k, MRR and citation accuracy are plain code
over gold spans — deterministic, free, and reproducible with no credentials. Relevance is span
containment, which is strict on purpose: a chunk about the right topic that lacks the span does
not count.

**Citation accuracy is reported as two numbers**, because the split is the diagnosis:

| | Value | Reading |
|---|---|---|
| Filing (accession) | **1.000** (n=21) | The top passage always came from the right filing. |
| Section (Item) | **0.680** (n=25) | It landed in the wrong section about a third of the time. |

It measures what an LLM judge is worst at — a fabricated citation looks exactly like a real one —
and needs no generator, since the top passage carries its own accession and Item. Cross-filing
questions pin no single accession, so that dimension is reported unscored for them rather than
guessed.

### The judge is calibrated before it is trusted

An uncalibrated judge produces figures that look like evidence without being evidence. Claude
Haiku 4.5 was measured against 50 balanced triples whose labels are certain **by construction** —
each a matched pair of a verified gold answer and the same answer with one fact corrupted against
the same passage:

**TPR 1.000 · TNR 1.000 · zero disagreements** ([full report](eval/judge_calibration_report.md))

Read that as *the instrument is not broken*, **not** *the instrument is perfect*. The corruptions
are unambiguous by construction (`$394,328` → `$994,328`), which is easier than the subtle cases
real generation produces. At n=50 a single flipped judgement moves either rate by four points.

The first run scored 0.960/0.960 with two disagreements, and **both were defects in the
calibration set, not judge errors** — a context-collection bug that left a comparison question
labelled GROUNDED on evidence that did not support it, and a corruption that mangled a *year* into
`9022`, which the judge fairly called a typo. Calibrating the instrument found faults in the
calibration rig first.

**No faithfulness score appears in the benchmark table.** Judge-scored metrics need a generator as
well as a judge, and no generator is configured here.

## The CI gate

`.github/workflows/ci.yaml` runs five jobs. `lint`, `unit` and `retrieval-gate` need **no secrets**
— so forks get real coverage rather than a skipped job. `compose-smoke` runs the literal
`docker compose up` on a clean runner, which is how the one-command promise above is enforced
rather than asserted.

Gating is **two-sided**, in [`eval/thresholds.yaml`](eval/thresholds.yaml). An absolute floor alone
lets quality rot down to the bar and stop there; a regression tolerance alone lets a bad baseline
become permanent. Floors are set from measured values — a floor above what the system achieves
makes `main` permanently red and teaches everyone to ignore the gate. Tables get their own floor so
the hard case cannot hide inside a healthy average, and `min_samples` catches a run whose
denominator silently shrank.

Verified in both directions: a deliberate regression produces **15 threshold breaches and exit 1**;
the baseline exits 0. Results are written to the GitHub Actions job summary, so the Actions tab is
itself the evidence.

No floor is set on citation-section accuracy. At a measured 0.680, any floor would ratify a known
weakness rather than constrain it.

## Known limitations

Ordered by what I would fix first.

1. **n=25 is too small.** Nothing below a four-point difference in this README is readable, and
   several conclusions are provisional because of it.
2. **The reranker demotes tables.** Measured, diagnosed, not yet fixed — a shortened rerank header
   is the first thing to try.
3. **The lexical arm over-recalls.** OR-relaxation fixed an arm that returned nothing and
   over-corrected; a minimum-matching-terms threshold sits between the extremes.
4. **`abstain_recall` is not yet meaningful.** Refusing is a generation-layer decision, and
   retrieval-only mode can only approximate it.
5. **Two filings, one fiscal year.** Cross-company and cross-year questions are thin.
6. **No generation metrics.** Faithfulness and answer relevancy need a generator; the harness is
   built and unrun.
7. **The chunking ablation row is missing.** Comparing structure-aware chunking against fixed
   512-token chunks needs a full re-ingest per configuration.

## Data and terms

Filings are fetched from [SEC EDGAR](https://www.sec.gov/edgar) at run time and **not
redistributed here** — only accession numbers are committed. Requests carry a declared
`User-Agent` with a contact address, as SEC requires, and are throttled to 5 req/s against a
published limit of 10. SEC filings are US government works in the public domain.

This is a retrieval system over public disclosures. It is not investment advice, and every answer
shows the passage it came from.

## Layout

```
db/migrations/     numbered raw SQL; no ORM, so no Alembic
db/queries/        hybrid_rrf.sql — the RRF query, commented at length
src/edgar_eval/
  ingest/          EDGAR client, partitioning, Item sectionising, chunking
  retrieve/        hybrid search, filters with corpus clamping, reranking
  graph/           LangGraph nodes, prompts, state
  eval/            retrieval metrics, citation accuracy, the judge
  api/             FastAPI + a single-file UI, no build step
eval/              gold set, calibration labels, thresholds, baseline, results
docs/measurements.md   every number in this README, with how it was produced
```

## Licence

MIT — see [LICENSE](LICENSE).

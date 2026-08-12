.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ─────────────────────────────────────────────
.PHONY: install
install: ## Sync dependencies with uv
	uv sync --all-groups

.env:
	@test -f .env || cp .env.example .env
	@echo "created .env from .env.example"

# ── Stack ───────────────────────────────────────────────────
.PHONY: up
up: .env ## Bring the stack up and follow the logs
	docker compose up --build -d
	@echo
	@echo "  postgres  → localhost:$${POSTGRES_PORT:-5442}"
	@echo

.PHONY: down
down: ## Stop the stack (keeps the volume)
	docker compose down

.PHONY: clean
clean: ## Stop the stack and delete the database volume
	docker compose down -v

.PHONY: logs
logs: ## Follow all logs
	docker compose logs -f

# ── Database ────────────────────────────────────────────────
.PHONY: migrate
migrate: ## Apply database migrations
	uv run python scripts/migrate.py

.PHONY: migrate-check
migrate-check: ## List pending migrations without applying them
	uv run python scripts/migrate.py --dry-run

.PHONY: psql
psql: ## Open a psql shell against the running database
	docker compose exec postgres psql -U $${POSTGRES_USER:-edgar} -d $${POSTGRES_DB:-edgar}

# ── Quality ─────────────────────────────────────────────────
.PHONY: fmt
fmt: ## Format with ruff
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: lint
lint: ## Lint and type-check
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy

.PHONY: test
test: ## Run unit tests (no services required)
	uv run pytest tests/unit

.PHONY: test-all
test-all: ## Run every test, including those needing postgres
	uv run pytest

.PHONY: check
check: lint test ## Lint, type-check and unit-test

# ── Corpus and evaluation ───────────────────────────────────
.PHONY: corpus
corpus: ## Ingest the evaluation corpus (AAPL + MSFT FY2023 10-K)
	uv run python scripts/ingest_cli.py --ticker AAPL MSFT --form 10-K --years 2023 2023

.PHONY: validate-goldset
validate-goldset: ## Check every gold span exists verbatim in the corpus
	uv run python scripts/validate_goldset.py

.PHONY: eval-retrieval
eval-retrieval: validate-goldset ## Retrieval metrics — deterministic, free, NO API KEY
	uv run python scripts/run_eval.py --retrieval-only

.PHONY: eval
eval: validate-goldset ## Full evaluation, including judge-scored metrics (needs ANTHROPIC_API_KEY)
	uv run python scripts/run_eval.py

.PHONY: gate
gate: ## Check the latest results against eval/thresholds.yaml
	uv run python scripts/check_thresholds.py \
		--results eval/results/latest.json \
		--baseline eval/baseline.json \
		--thresholds eval/thresholds.yaml

.PHONY: baseline
baseline: ## Record the current results as the new baseline (review the diff!)
	cp eval/results/latest.json eval/baseline.json
	@echo "baseline updated — commit it deliberately, never as a drive-by"

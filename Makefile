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

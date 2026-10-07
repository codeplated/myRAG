# Common commands. Run `make help` to list them.
export PYTHONPATH := src

.PHONY: help install up down run ingest eval eval-answers test test-db lint

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install: ## Install the app and the development tools into the current Python environment
	pip install -r requirements-dev.txt

up: ## Start the database and the app in Docker (http://localhost:8000)
	docker compose up --build

down: ## Stop the Docker containers
	docker compose down

run: ## Run the app locally with auto-reload (needs the database: docker compose up -d pgvector)
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

ingest: ## Ingest every PDF in BOOKS_DIR
	python -m rag_pipeline.ingest_books

eval: ## Compare the search configurations on the test questions
	python -m rag_pipeline.evaluate

eval-answers: ## Same as eval, and also judge the faithfulness of the answers (slower)
	python -m rag_pipeline.evaluate --answers

test: ## Run the unit tests (no database, no network, no API keys needed)
	pytest

test-db: ## Run the database tests against a throwaway database called rag_test
	RAG_TEST_DATABASE=1 pytest -m integration

lint: ## Check code style
	ruff check .

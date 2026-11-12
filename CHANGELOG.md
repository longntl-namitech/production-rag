# Changelog

## v0.2.0 - 2026-11-12

The repo is renamed from `retrieval-playbook` to `production-rag` and restructured from a retrieval service into a lab of RAG methods. Old GitHub URLs redirect.

### Changed

- Package `retrieval_playbook` is now `production_rag` (no compatibility shim), and the distribution is `production-rag`.
- New layout: `core/` plus `methods/<name>/` (README, pipeline, example). The `docs/<method>.md` pages became the method READMEs.
- The default install is light: `torch` and `sentence-transformers` moved to the `local-models` extra. A new `parse` extra holds the document parsers.
- `docker-compose.yml` runs Qdrant only.

### Added

- `naive_rag` baseline.
- OpenAI chat client with a token budget.
- Answer eval on rag-mini-wikipedia and the vi_RAG LegalRAG config, with an LLM judge.
- `deadline_fanout` latency bench.
- `parse_and_chunk` with pip-only parsers, recursive and Markdown-heading chunking, a parsing-quality eval and a chunking comparison.
- `fact_check` with its SciFact claim eval.

### Removed

- FastAPI service and bearer auth.
- Root `Dockerfile` and the `retriever` compose service.
- Contract tests.

### Fixed

- `rerank_min_budget_ms` under 50 reported `failed`.
- A reranker returning the wrong number of scores.
- Rerank with no reranker widened the candidate pool.
- Vietnamese BM25 was not warmed up before timing.
- `percentile` with one value.
- A failed baseline query counted as a rescue.
- The report hid failure counts.
- `ensure_collection` skipped the dimension check.
- Context expansion stopped at 1,000 chunks.

Still open: see `BACKLOG.md`.

## v0.1.0 - 2026-10-30

First release: hybrid retrieval service with eval on SciFact and TVPL.

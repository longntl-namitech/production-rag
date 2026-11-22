# Changelog

## Unreleased

### Added

- `retrieval_sufficiency`: after retrieval an LLM says whether the top 5 passages are enough to answer the question; own eval against a "gold document in the top 5" proxy.
- `llm_rerank`: an LLM scores the top 20 dense results from 0 to 10 and the best 5 are kept; measured in the retrieval eval (`core/eval/run_llm.py`, rows `naive_rag` and `llm_rerank`).

## v0.3.0 - 2026-11-16

First batch of techniques from the [RAG_Techniques](https://github.com/nirdiamant/RAG_Techniques) catalog.

### Added

- `hyde`: an LLM-written hypothetical answer passage replaces the question for dense search.
- `step_back`: retrieval with the question and a broader step-back question.
- `query_decomposition`: a question split into subqueries, merged with `multi_query`.
- Semantic chunking in `parse_and_chunk` (`chunk_semantic`): chunks of up to 800 characters, cut where the topic changes.
- `contextual_headers`: an LLM-written sentence that places each chunk in its document, embedded and BM25-encoded with the chunk at index time; the stored text stays the raw chunk.
- `contextual_compression`: an LLM keeps only the retrieved sentences relevant to the question; measured in the answer eval only.
- Query-side LLM helper (`core/query_llm.py`): an unusable output falls back to the original question.
- On-disk cache for LLM text (`core/cache.py`), git-ignored by default.
- Retrieval eval for methods that call an LLM (`core/eval/run_llm.py`), with a token budget and `--estimate-only`.
- README results chart and stage-grouped methods table.

### Fixed

- LLM eval runs that never reached the LLM were not stopped.
- A contextual headers collection with missing context sentences was kept; `index` now raises `ContextIncomplete` and writes nothing.
- The contextual headers sentence cap was 100 tokens, which cut about 1 in 5 Vietnamese context sentences; it is now 200.

## v0.2.0 - 2026-11-12

The repo is renamed from `retrieval-playbook` to `production-rag` and restructured from a retrieval service into a lab of RAG methods. Old GitHub URLs redirect.

### Changed

- Package `retrieval_playbook` is now `production_rag` (no compatibility shim), and the distribution is `production-rag`.
- New layout: `core/` plus `methods/<name>/` (README, pipeline, example). The `docs/<method>.md` pages were replaced by the method READMEs.
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

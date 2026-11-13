# HyDE: search with a hypothetical answer

Extra: `local-models`. Needs an LLM at query time (`OPENAI_API_KEY`).

Differs from [`multi_query`](../multi_query/README.md): one hypothetical answer passage replaces the question, instead of several rephrasings of it.

## When to use
- Questions are short or phrased very differently from the documents, so the question's own vector sits far from the passages that answer it.

## When not to
- Hybrid retrieval already finds the answer, or the model does not know the domain: an invented passage with wrong facts pulls the search the wrong way.

## How it works
- [`generate.py`](generate.py): the LLM writes a short passage that would answer the question (JSON, capped at 256 tokens). Any bad output (not JSON, cut at the cap, a failed call) gives the original question, so search still runs. `BudgetExceeded` propagates.
- [`pipeline.py`](pipeline.py): `search(queries, ...)` runs the passage through the dense vectors only. A passage matches by meaning, not by words, so the sparse leg is skipped for every query in the list. Because the search type applies to the whole list, with `include_question=True` the question is dense-only too.
- Two ways to build the queries, both in the table: `hyde_passage` (the passage alone) and `hyde_passage_query` (the question and the passage, merged by best score as in `multi_query`).

Source: Gao et al., [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (2022). Own prompt, reimplemented from the paper.

## Run

```bash
OPENAI_API_KEY=... uv run --extra local-models python -m production_rag.methods.hyde.example
```

Without a key the example searches the question and says so.

## Results
Not measured yet.

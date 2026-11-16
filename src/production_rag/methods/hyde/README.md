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
Run 2026-11-13, `gpt-5.4-nano`, one run per dataset with all the LLM query rows together (318,658 tokens on SciFact, 1,280,815 on TVPL). The same baseline moved by about 0.002 nDCG@10 between two runs with no code change, so smaller differences are noise. Full tables: [SciFact](../../../../results/query_side_scifact.md), [TVPL](../../../../results/query_side_tvpl.md).

SciFact (300 queries):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 134 | 451 |
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.739 | 0.770 | 148 | 209 |
| `hyde_passage` | 0.903 | 0.973 | 0.724 | 0.764 | 1447 | 4168 |
| `hyde_passage_query` | 0.894 | 0.970 | 0.725 | 0.762 | 1437 | 2776 |

TVPL (1000 queries, Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 64 | 88 |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 68 | 78 |
| `hyde_passage` | 0.925 | 0.987 | 0.782 | 0.809 | 1805 | 2359 |
| `hyde_passage_query` | 0.935 | 0.990 | 0.802 | 0.826 | 1836 | 2383 |

On TVPL, `hyde_passage_query` is the best row, +0.018 nDCG@10 over `hybrid_weighted_rrf`. On SciFact neither HyDE row beats `hybrid_weighted_rrf`. Both add an LLM call, about 1.4 to 1.8 s at p50.

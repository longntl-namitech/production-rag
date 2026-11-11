# Multi-query merge

Extra: `local-models`

## When to use
- A question has several parts, or a rewriter produces several phrasings, and each should retrieve on its own.

## How it works
- `search` takes `queries`, a list of up to 8 subqueries. All are embedded in one call per vector kind ([`pipeline/search.py`](../../core/pipeline/search.py)).
- Each spec searches every subquery, then [`merge_max`](../../core/pipeline/merge.py) keeps each chunk once with its best score.
- When reranking, the `rerank_query` argument (or the first subquery) is used to score the merged pool, so the reranker judges against the original question rather than one fragment.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.multi_query.example
```

## Results
No public numbers yet. The public datasets have one query per question; producing subqueries needs an LLM rewriter, which is on the roadmap. Behaviour is covered by tests (`test_multi_query_merged`).

## Lessons from production
- Measured on internal data, data not published. Overlapping retrieval with the query rewrite (speculative search) saved about 1 s per turn. That method needs an LLM and is on the roadmap.

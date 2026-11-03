# Dense retrieval

Extra: `local-models`

## When to use
- Queries are phrased differently from the documents (paraphrases, questions against statements).
- You need one method that is good across domains and languages.

## When not to
- Exact identifiers, codes, article numbers or rare names matter most. Dense models blur them; BM25 keeps them.
- You cannot afford an embedding call per query (it is the slowest leg here, ~20 ms on CPU per query with a 212M model).

## How it works
- [`embedders/jina_local.py`](../../core/embedders/jina_local.py): `jina-embeddings-v5-text-nano-retrieval` with separate `query` and `document` prompts, 768 dimensions, normalised, cosine in Qdrant. Loaded in float32 because bfloat16 weights run slowly on most CPUs.
- [`embedders/openai_compat.py`](../../core/embedders/openai_compat.py): any OpenAI-compatible `/embeddings` endpoint instead, set `EMBEDDING_BASE_URL`.
- The query is embedded once per request, whatever the number of specs ([`pipeline/search.py`](../../core/pipeline/search.py)).

## Run

```bash
uv run --extra local-models python -m production_rag.methods.dense.example
```

## Results
SciFact:

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 21 | 26 |  |  |

TVPL (Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |

On TVPL, dense alone is the best method in this repo (see [hybrid fusion](../hybrid_fusion/README.md)).

## Lessons from building this repo
- Warm every model before the first timed query (`warm_up`). Otherwise the first query spends its whole 4 s deadline loading weights.

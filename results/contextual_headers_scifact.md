# scifact

100 queries, 350 documents, run 2026-11-16. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `chunk_recursive_800` | 0.960 | 1.000 | 0.917 | 0.928 | 57 | 76 |  |  |
| `contextual_headers` | 0.960 | 1.000 | 0.905 | 0.919 | 56 | 62 |  |  |

No failed queries.

# tvpl

1000 queries, 10576 documents, run 2026-11-22. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.917 | 0.977 | 0.794 | 0.815 | 82 | 150 |  |  |
| `llm_rerank` | 0.928 | 0.938 | 0.835 | 0.849 | 1892 | 2422 |  |  |

No failed queries.

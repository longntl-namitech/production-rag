# scifact

300 queries, 5183 documents, run 2026-11-13. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.736 | 0.769 | 67 | 84 |  |  |
| `query_decomposition` | 0.892 | 0.973 | 0.736 | 0.770 | 1396 | 1987 |  |  |

No failed queries.

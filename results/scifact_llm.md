# scifact

300 queries, 5183 documents, run 2026-11-22. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.875 | 0.962 | 0.715 | 0.751 | 81 | 156 |  |  |
| `llm_rerank` | 0.877 | 0.883 | 0.736 | 0.768 | 1955 | 2564 |  |  |

No failed queries.

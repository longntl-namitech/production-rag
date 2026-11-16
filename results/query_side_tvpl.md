# tvpl

1000 queries, 10576 documents, run 2026-11-13. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 64 | 88 |  |  |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 68 | 78 |  |  |
| `multi_query` | 0.918 | 0.980 | 0.787 | 0.809 | 1394 | 1729 |  |  |
| `hyde_passage` | 0.925 | 0.987 | 0.782 | 0.809 | 1805 | 2359 |  |  |
| `hyde_passage_query` | 0.935 | 0.990 | 0.802 | 0.826 | 1836 | 2383 |  |  |
| `step_back` | 0.916 | 0.978 | 0.780 | 0.803 | 1272 | 1576 |  |  |
| `query_decomposition` | 0.906 | 0.980 | 0.759 | 0.786 | 1641 | 2052 |  |  |

No failed queries.

# scifact

300 queries, 5183 documents, run 2026-11-13. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 134 | 451 |  |  |
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.739 | 0.770 | 148 | 209 |  |  |
| `multi_query` | 0.879 | 0.973 | 0.745 | 0.775 | 1374 | 1939 |  |  |
| `hyde_passage` | 0.903 | 0.973 | 0.724 | 0.764 | 1447 | 4168 |  |  |
| `hyde_passage_query` | 0.894 | 0.970 | 0.725 | 0.762 | 1437 | 2776 |  |  |
| `step_back` | 0.907 | 0.982 | 0.741 | 0.778 | 1217 | 1452 |  |  |
| `query_decomposition` | 0.886 | 0.958 | 0.716 | 0.753 | 1663 | 3640 |  |  |

Failed queries: `query_decomposition` 4.

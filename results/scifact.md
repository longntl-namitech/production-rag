# scifact

300 queries, 5183 documents, run 2026-10-30. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 21 | 26 |  |  |
| `bm25_no_idf` | 0.714 | 0.859 | 0.572 | 0.601 | 6 | 8 |  |  |
| `bm25` | 0.809 | 0.921 | 0.647 | 0.683 | 7 | 8 |  |  |
| `bm25_vi_stopwords` | 0.793 | 0.886 | 0.627 | 0.661 | 7 | 8 |  |  |
| `hybrid_rrf` | 0.882 | 0.973 | 0.718 | 0.755 | 23 | 28 |  |  |
| `hybrid_dbsf` | 0.880 | 0.957 | 0.716 | 0.752 | 23 | 27 |  |  |
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.736 | 0.769 | 23 | 26 |  |  |
| `hybrid_rrf_rerank` | 0.752 | 0.973 | 0.636 | 0.660 | 985 | 1096 | 6.0% | 11.0% |

# tvpl

1000 queries, 10576 documents, run 2026-10-30. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |
| `bm25_no_idf` | 0.751 | 0.873 | 0.613 | 0.636 | 8 | 9 |  |  |
| `bm25` | 0.800 | 0.915 | 0.641 | 0.669 | 7 | 9 |  |  |
| `bm25_vi_stopwords` | 0.768 | 0.910 | 0.617 | 0.645 | 7 | 11 |  |  |
| `hybrid_rrf` | 0.912 | 0.974 | 0.757 | 0.786 | 26 | 36 |  |  |
| `hybrid_dbsf` | 0.899 | 0.963 | 0.773 | 0.793 | 25 | 32 |  |  |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 26 | 31 |  |  |
| `hybrid_rrf_rerank` | 0.906 | 0.974 | 0.792 | 0.810 | 812 | 1013 | 11.0% | 4.9% |

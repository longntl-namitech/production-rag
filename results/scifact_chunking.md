# scifact

300 queries, 5183 documents, run 2026-11-12. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

Chunked variants search their own collections; a document counts once, at its best chunk, so recall@100 counts the documents in the top 100 chunks.

| Collection | Chunker | Size | Overlap | Chunks | Mean chars |
|---|---|---|---|---|---|
| `eval-scifact-c800` | fixed | 800 | 0 | 11805 | 615 |
| `eval-scifact-r800` | recursive | 800 | 0 | 12315 | 589 |
| `eval-scifact-r800o200` | recursive | 800 | 200 | 16143 | 583 |
| `eval-scifact-r400` | recursive | 400 | 0 | 24724 | 293 |

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.875 | 0.962 | 0.715 | 0.751 | 20 | 27 |  |  |
| `chunk_recursive_800` | 0.878 | 0.958 | 0.715 | 0.750 | 22 | 27 |  |  |
| `chunk_recursive_800_o200` | 0.870 | 0.948 | 0.717 | 0.750 | 22 | 28 |  |  |
| `chunk_recursive_400` | 0.871 | 0.947 | 0.706 | 0.743 | 21 | 27 |  |  |

No failed queries.

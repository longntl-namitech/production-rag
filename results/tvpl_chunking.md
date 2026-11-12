# tvpl

1000 queries, 10576 documents, run 2026-11-12. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

Chunked variants search their own collections; a document counts once, at its best chunk, so recall@100 counts the documents in the top 100 chunks.

| Collection | Chunker | Size | Overlap | Chunks | Mean chars |
|---|---|---|---|---|---|
| `eval-tvpl-c800` | fixed | 800 | 0 | 17726 | 559 |
| `eval-tvpl-r800` | recursive | 800 | 0 | 18556 | 534 |
| `eval-tvpl-r800o200` | recursive | 800 | 200 | 23813 | 525 |
| `eval-tvpl-r400` | recursive | 400 | 0 | 35277 | 281 |

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.917 | 0.977 | 0.793 | 0.815 | 20 | 28 |  |  |
| `chunk_recursive_800` | 0.914 | 0.976 | 0.787 | 0.810 | 22 | 28 |  |  |
| `chunk_recursive_800_o200` | 0.909 | 0.970 | 0.781 | 0.802 | 22 | 26 |  |  |
| `chunk_recursive_400` | 0.905 | 0.969 | 0.770 | 0.794 | 22 | 26 |  |  |

No failed queries.

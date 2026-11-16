# tvpl

1000 queries, 10576 documents, run 2026-11-13. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

Chunked variants search their own collections; a document counts once, at its best chunk, so recall@100 counts the documents in the top 100 chunks.

| Collection | Chunker | Size | Overlap | Chunks | Mean chars |
|---|---|---|---|---|---|
| `eval-tvpl-r800` | recursive | 800 | 0 | 18556 | 534 |
| `eval-tvpl-s800` | semantic | 800 | 0 | 22027 | 450 |

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `chunk_recursive_800` | 0.914 | 0.976 | 0.786 | 0.809 | 24 | 28 |  |  |
| `chunk_semantic` | 0.908 | 0.972 | 0.778 | 0.801 | 23 | 27 |  |  |

No failed queries.

# scifact

300 queries, 5183 documents, run 2026-11-13. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

Chunked variants search their own collections; a document counts once, at its best chunk, so recall@100 counts the documents in the top 100 chunks.

| Collection | Chunker | Size | Overlap | Chunks | Mean chars |
|---|---|---|---|---|---|
| `eval-scifact-r800` | recursive | 800 | 0 | 12315 | 589 |
| `eval-scifact-s800` | semantic | 800 | 0 | 14618 | 496 |

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `chunk_recursive_800` | 0.878 | 0.958 | 0.715 | 0.750 | 22 | 48 |  |  |
| `chunk_semantic` | 0.869 | 0.958 | 0.717 | 0.750 | 20 | 26 |  |  |

No failed queries.

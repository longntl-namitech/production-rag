# tvpl

1000 queries, 10576 documents, run 2026-11-11. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

`naive_rag` searches 17726 chunks of up to 800 characters; a document counts once, at its best chunk, so recall@100 counts the documents in the top 100 chunks.

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.917 | 0.977 | 0.793 | 0.815 | 23 | 28 |  |  |

No failed queries.

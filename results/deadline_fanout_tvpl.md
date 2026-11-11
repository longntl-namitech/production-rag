200 queries, base `hybrid_rrf`, specs `dense`, `sparse`, `hybrid_rerank`, CPU: arm, 10 cores, one query at a time.

| Deadline ms | p50 ms | p95 ms | All specs ok | Partial | Expired | Rerank ok | Rerank skipped |
|---|---|---|---|---|---|---|---|
| 2000 | 366 | 382 | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 500 | 364 | 377 | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 150 | 22 | 26 | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| sequential | 377 | 390 | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |

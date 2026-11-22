# Retrieval sufficiency on tvpl

1000 queries, 10,576 documents, base `hybrid_weighted_rrf`, k=5, model `gpt-5.4-nano`, 2026-11-23, one run. The label is the model's own judgement. Agreement with "a gold document is in the top k" is a proxy for "enough to answer", not a measure of answerability. Failed items are excluded.

Agreement 0.862 on 1000 answered queries, 0 failed. `sufficient` on 84.3% of them; a gold document is in the top 5 for 90.3%.

| Label | Gold in top k | Gold not in top k |
|---|---|---|
| sufficient | 804 | 39 |
| insufficient | 99 | 58 |

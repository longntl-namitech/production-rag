# Retrieval sufficiency on scifact

300 queries, 5,183 documents, base `hybrid_weighted_rrf`, k=5, model `gpt-5.4-nano`, 2026-11-23, one run. The label is the model's own judgement. Agreement with "a gold document is in the top k" is a proxy for "enough to answer", not a measure of answerability. Failed items are excluded.

Agreement 0.513 on 300 answered queries, 0 failed. `sufficient` on 41.7% of them; a gold document is in the top 5 for 84.3%.

| Label | Gold in top k | Gold not in top k |
|---|---|---|
| sufficient | 116 | 9 |
| insufficient | 137 | 38 |

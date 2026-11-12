# Fact check on SciFact claims

300 claims from the dev split, 5,183 abstracts, base `hybrid_weighted_rrf`, k=5, model `gpt-5.4-nano`, 2026-11-12, one run. Failed items are excluded from accuracy, precision and recall.

| Label | Gold | Predicted | Correct | Precision | Recall |
|---|---|---|---|---|---|
| supported | 124 | 118 | 93 | 0.788 | 0.750 |
| refuted | 64 | 39 | 24 | 0.615 | 0.375 |
| unverified | 112 | 143 | 94 | 0.657 | 0.839 |

Accuracy 0.703 on 300 claims, 0 failed, unverified 47.7%, gold evidence in top 5: 95.7% (accuracy 0.639 when found, 0.250 when not).

| Gold \ predicted | supported | refuted | unverified | failed |
|---|---|---|---|---|
| supported | 93 | 7 | 24 | 0 |
| refuted | 15 | 24 | 25 | 0 |
| unverified | 10 | 8 | 94 | 0 |

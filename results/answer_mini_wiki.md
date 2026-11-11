# mini_wiki

150 sampled questions, 3200 documents in 3613 chunks of up to 800 characters, run 2026-11-12. Generator and judge `gpt-5.4-nano`, top 5 passages. Judged by the same model that answers.

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.803 | 69.3% | 8.7% | 30 |  | 942 | 632 | 757 | 0 | 0 |
| `hybrid_fusion` | 0.833 | 69.3% | 8.7% | 32 |  | 844 | 657 | 797 | 0 | 0 |
| `rerank` | 0.837 | 63.3% | 6.7% | 131 |  | 846 | 617 | 736 | 0 | 0 |
| `multi_query` | 0.797 | 64.0% | 10.0% | 40 | 1208 | 870 | 831 | 795 | 0 | 0 |
| `context_expansion` | 0.863 | 68.7% | 6.7% | 36 |  | 850 | 744 | 931 | 0 | 0 |

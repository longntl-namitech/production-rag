# mini_wiki

150 sampled questions, 3200 documents in 3613 chunks of up to 800 characters, run 2026-11-12. Generator and judge `gpt-5.4-nano`, top 5 passages. Judged by the same model that answers.

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.797 | 97.3% | 8.7% | 30 |  | 900 | 633 | 757 | 0 | 0 |
| `hybrid_fusion` | 0.813 | 96.7% | 8.7% | 31 |  | 837 | 655 | 793 | 0 | 0 |
| `rerank` | 0.823 | 96.7% | 8.7% | 137 |  | 821 | 617 | 735 | 0 | 0 |
| `multi_query` | 0.817 | 96.0% | 8.0% | 41 | 1128 | 804 | 834 | 801 | 0 | 0 |
| `context_expansion` | 0.830 | 96.0% | 8.7% | 35 |  | 804 | 744 | 932 | 0 | 0 |

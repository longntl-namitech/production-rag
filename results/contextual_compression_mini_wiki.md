# mini_wiki

50 sampled questions, 3200 documents in 3613 chunks of up to 800 characters, run 2026-11-16. Generator and judge `gpt-5.4-nano`, top 5 passages. Judged by the same model that answers.

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.810 | 98.0% | 12.0% | 28 |  | 885 | 641 | 769 | 0 | 0 |
| `context_expansion` | 0.860 | 100.0% | 10.0% | 32 |  | 890 | 764 | 957 | 0 | 0 |
| `contextual_compression` | 0.800 | 98.0% | 10.0% | 996 |  | 753 | 963 | 96 | 0 | 0 |

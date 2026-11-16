# vi_legal

40 sampled questions, 2125 documents in 5788 chunks of up to 800 characters, run 2026-11-16. Generator and judge `gpt-5.4-nano`, top 5 passages. Judged by the same model that answers.

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.575 | 97.5% | 17.5% | 32 |  | 1357 | 1228 | 1186 | 0 | 0 |
| `context_expansion` | 0.662 | 100.0% | 17.5% | 34 |  | 1273 | 2526 | 2651 | 0 | 0 |
| `contextual_compression` | 0.562 | 100.0% | 15.0% | 1054 |  | 1153 | 1811 | 336 | 0 | 0 |

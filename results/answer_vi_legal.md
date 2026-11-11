# vi_legal

100 sampled questions, 2125 documents in 5788 chunks of up to 800 characters, run 2026-11-12. Generator and judge `gpt-5.4-nano`, top 5 passages. Judged by the same model that answers.

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.620 | 98.0% | 18.0% | 34 |  | 1224 | 1200 | 1153 | 0 | 0 |
| `hybrid_fusion` | 0.580 | 95.0% | 23.0% | 38 |  | 1260 | 1213 | 1169 | 0 | 0 |
| `rerank` | 0.630 | 96.0% | 15.0% | 156 |  | 1154 | 1193 | 1136 | 0 | 0 |
| `multi_query` | 0.610 | 97.0% | 18.0% | 66 | 1372 | 1190 | 1436 | 1171 | 0 | 0 |
| `context_expansion` | 0.685 | 92.0% | 15.0% | 37 |  | 1209 | 2455 | 2570 | 0 | 0 |

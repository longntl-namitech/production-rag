# vi_legal

100 sampled questions, 2125 documents in 5788 chunks of up to 800 characters, run 2026-11-12. Generator and judge `gpt-5.4-nano`, top 5 passages. Judged by the same model that answers.

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.605 | 89.0% | 19.0% | 34 |  | 1270 | 1197 | 1153 | 0 | 0 |
| `hybrid_fusion` | 0.620 | 90.0% | 16.0% | 35 |  | 1232 | 1214 | 1170 | 0 | 0 |
| `rerank` | 0.620 | 96.0% | 14.0% | 142 |  | 1231 | 1193 | 1135 | 0 | 0 |
| `multi_query` | 0.645 | 88.0% | 16.0% | 64 | 1452 | 1196 | 1442 | 1172 | 0 | 0 |
| `context_expansion` | 0.615 | 88.0% | 19.0% | 39 |  | 1358 | 2452 | 2575 | 0 | 0 |

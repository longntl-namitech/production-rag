# Reranking with a cross-encoder

Extra: `local-models`

## When to use
- Precision at the top matters (one answer shown, or few chunks sent to an LLM) and you can spend about a second per query, or have a GPU or a hosted reranker.
- The reranker was trained on data that looks like yours, in your language.

## When not to
- The reranker is out of domain. On SciFact the multilingual mMiniLM cross-encoder makes results worse.
- The deadline is tight. Here it adds ~0.8 to 1 s p50 on CPU for a pool of 100.

## How it works
- [`pipeline/rerank_stage.py`](../../core/pipeline/rerank_stage.py):
  - The candidate pool is widened to `min(k * 4, 100)` before reranking, then cut back to `k`.
  - Skipped when less than 300 ms of the request budget is left (`rerank: skipped_budget`), or when there is one candidate (`skipped_trivial`).
  - On error or timeout the retrieval order is kept (`rerank: failed`); the spec still succeeds.
- Backends ([`rerankers/`](../../core/rerankers/)): local `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, or any Cohere-compatible `/rerank` endpoint (`RERANK_BASE_URL`, `RERANK_API_KEY`).
- Rescue: the top result was wrong before reranking and is right after. Damage: the reverse. Both are measured against `hybrid_rrf` on the same query.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.rerank.example
```

## Results
SciFact:

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rrf` | 0.882 | 0.973 | 0.718 | 0.755 | 23 | 28 |  |  |
| `hybrid_rrf_rerank` | 0.752 | 0.973 | 0.636 | 0.660 | 985 | 1096 | 6.0% | 11.0% |

TVPL (Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rrf` | 0.912 | 0.974 | 0.757 | 0.786 | 26 | 36 |  |  |
| `hybrid_rrf_rerank` | 0.906 | 0.974 | 0.792 | 0.810 | 812 | 1013 | 11.0% | 4.9% |

- On TVPL reranking lifts MRR@10 from 0.757 to 0.792 (rescue 11.0%, damage 4.9%), but dense alone is about as good (0.797) at a fraction of the cost.
- On SciFact it damages more than it rescues (11.0% vs 6.0%) and lowers nDCG@10 from 0.755 to 0.660. The reranker scores title plus text, the same text the other legs indexed.

### Answer quality

Run 2026-11-12, `gpt-5.4-nano`, top 5 passages. Full tables: [mini_wiki](../../../../results/answer_mini_wiki.md), [vi_legal](../../../../results/answer_vi_legal.md).

mini_wiki (150 questions, English):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.803 | 69.3% | 8.7% | 30 |  | 942 | 632 | 757 | 0 | 0 |
| `rerank` | 0.837 | 63.3% | 6.7% | 131 |  | 846 | 617 | 736 | 0 | 0 |

vi_legal (100 questions, Vietnamese legal):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.605 | 89.0% | 19.0% | 34 |  | 1270 | 1197 | 1153 | 0 | 0 |
| `rerank` | 0.620 | 96.0% | 14.0% | 142 |  | 1231 | 1193 | 1135 | 0 | 0 |

Answer quality is judged by the same model that writes the answers (`gpt-5.4-nano`), which can flatter its own answers. All methods share the generator and the judge, so the rows can be compared with each other. Abstentions count as incorrect and faithful, so a method that abstains more can score higher on faithfulness. Correctness and faithfulness are averaged over the questions the judge scored, so the denominators can differ when a judge call fails (the `Judge failed` column). `Passage tokens (est.)` is estimated as characters / 3. The `vi_RAG` gold answers were aligned with LLMs per the dataset card (Mixtral 8x22B, GPT-4).

## Lessons from production
- Measured on internal data, data not published. The same cross-encoder over a pool of 20 raised accuracy@1 from 30.9% to 63.8% and MRR@10 from 0.46 to 0.74, rescued 37% of queries and damaged 4.2%, at 213 ms p50 on CPU. On a second set, a hosted Cohere reranker scored 8.2 points above it.
- Measured on internal data, data not published. Hybrid plus rerank lifted recall@10 from 0.416 to 0.462, and p50 latency from 506 ms to 1,530 ms.
- Measured on internal data, data not published. LLM-based and small decision-model rerankers, pointwise and listwise, scored below no reranking at all.
- Measured on internal data, data not published. About 22% of the remaining misses traced back to label noise in the test set, which sets a ceiling no reranker can pass.
- Treat reranking as optional at runtime: skip it when the budget is nearly spent and fall back to retrieval order on any error, so a slow reranker never fails a request.

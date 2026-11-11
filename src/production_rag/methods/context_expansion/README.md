# Context expansion

Extra: `local-models`

## When to use
- Chunks are small and the answer spans the chunk boundary, or a short FAQ entry should come back whole.

## How it works
- `search` takes a `context` argument, `ContextOption(prev=n, next=m)` or `ContextOption(full=True)` ([`pipeline/context.py`](../../core/pipeline/context.py)).
- Chunks are stored with `doc_id` and `chunk_index`; neighbours are fetched with one filtered scroll per hit, after ranking, so they never affect scores.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.context_expansion.example
```

## Results
Expansion runs after ranking, so the retrieval ranking cannot change and there is no retrieval row. The answer eval measures it through the passages sent to the generator. The answer eval indexes 800-character chunks: mini_wiki has 3200 documents in 3613 chunks and vi_legal has 2125 documents in 5788 chunks, so vi_legal documents split far more often. Behaviour is also covered by tests (`test_expand_prev_next`, `test_expand_full_document`, `test_context_expansion_applied`).

Run 2026-11-12, `gpt-5.4-nano`, top 5 passages. Full tables: [mini_wiki](../../../../results/answer_mini_wiki.md), [vi_legal](../../../../results/answer_vi_legal.md).

mini_wiki (150 questions, English):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_fusion` | 0.833 | 69.3% | 8.7% | 32 |  | 844 | 657 | 797 | 0 | 0 |
| `context_expansion` | 0.863 | 68.7% | 6.7% | 36 |  | 850 | 744 | 931 | 0 | 0 |

vi_legal (100 questions, Vietnamese legal):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_fusion` | 0.620 | 90.0% | 16.0% | 35 |  | 1232 | 1214 | 1170 | 0 | 0 |
| `context_expansion` | 0.615 | 88.0% | 19.0% | 39 |  | 1358 | 2452 | 2575 | 0 | 0 |

Answer quality is judged by the same model that writes the answers (`gpt-5.4-nano`), which can flatter its own answers. All methods share the generator and the judge, so the rows can be compared with each other. Abstentions count as incorrect and faithful, so a method that abstains more can score higher on faithfulness. Correctness and faithfulness are averaged over the questions the judge scored, so the denominators can differ when a judge call fails (the `Judge failed` column). `Passage tokens (est.)` is estimated as characters / 3. The `vi_RAG` gold answers were aligned with LLMs per the dataset card (Mixtral 8x22B, GPT-4).

# Multi-query merge

Extra: `local-models`

## When to use
- A question has several parts, or a rewriter produces several phrasings, and each should retrieve on its own.

## How it works
- `search` takes `queries`, a list of up to 8 subqueries. All are embedded in one call per vector kind ([`pipeline/search.py`](../../core/pipeline/search.py)).
- Each spec searches every subquery, then [`merge_max`](../../core/pipeline/merge.py) keeps each chunk once with its best score.
- When reranking, the `rerank_query` argument (or the first subquery) is used to score the merged pool, so the reranker judges against the original question rather than one fragment. `rerank_query` only applies with a base that reranks (`base="hybrid_rrf_rerank"`); with the default base `search` raises `ValueError`.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.multi_query.example
```

## Results
There is no retrieval row. The retrieval datasets have one query per question and the subqueries come from an LLM rewrite, so the method is only measured through the answer eval. `Rewrite p50 ms` is the rewrite call; `RAG tokens / question` includes the rewrite and generation calls. Behaviour is also covered by tests (`test_multi_query_merged`).

Run 2026-11-12, `gpt-5.4-nano`, top 5 passages. Full tables: [mini_wiki](../../../../results/answer_mini_wiki.md), [vi_legal](../../../../results/answer_vi_legal.md).

mini_wiki (150 questions, English):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_fusion` | 0.813 | 96.7% | 8.7% | 31 |  | 837 | 655 | 793 | 0 | 0 |
| `multi_query` | 0.817 | 96.0% | 8.0% | 41 | 1128 | 804 | 834 | 801 | 0 | 0 |

vi_legal (100 questions, Vietnamese legal):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_fusion` | 0.580 | 95.0% | 23.0% | 38 |  | 1260 | 1213 | 1169 | 0 | 0 |
| `multi_query` | 0.610 | 97.0% | 18.0% | 66 | 1372 | 1190 | 1436 | 1171 | 0 | 0 |

Answer quality is judged by the same model that writes the answers (`gpt-5.4-nano`), which can flatter its own answers. All methods share the generator and the judge. Abstentions count as incorrect and faithful, so a method that abstains more can score higher on faithfulness. Correctness and faithfulness are averaged over the questions the judge scored, so the denominators can differ when a judge call fails (the `Judge failed` column). `Passage tokens (est.)` is estimated as characters / 3. The `vi_RAG` gold answers were aligned with LLMs per the dataset card (Mixtral 8x22B, GPT-4). The rag-mini-wikipedia data is `cc-by-3.0` per its dataset card.

The answer eval was run twice on 2026-11-12; the first run had a faithfulness-judge bug, since fixed, and the tables here are from the second run. Correctness moved by up to 0.07 between the two runs with no change to the correctness judge, so differences smaller than that are within run-to-run variance and should not be read as one method beating another.

## Lessons from production
- Measured on internal data, data not published. Overlapping retrieval with the query rewrite (speculative search) saved about 1 s per turn. That method needs an LLM and is on the roadmap.

# Naive RAG

Extra: `local-models`

LLM: `OPENAI_API_KEY` (model `OPENAI_MODEL`, default `gpt-5.4-nano`)

## When to use
- It is the baseline every other method is compared against.

## How it works
- Documents are split into chunks of up to 800 characters by [`chunking.py`](../../core/chunking.py).
- The query is embedded and the top 5 chunks come back from dense search.
- The passages are stuffed into the prompt in [`generation.py`](../../core/generation.py), which replies `NO_ANSWER` when the passages lack the answer.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.naive_rag.example
```

Without `OPENAI_API_KEY` the example prints the retrieved chunks and stops before the generation step.

## Results
Retrieval, run 2026-11-11 ([SciFact](../../../../results/scifact_naive_rag.md), [TVPL](../../../../results/tvpl_naive_rag.md)). It searches 800-character chunks and a document counts once, at its best chunk, so recall@100 counts the documents in the top 100 chunks.

SciFact:

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.875 | 0.962 | 0.715 | 0.751 | 22 | 26 |  |  |

TVPL (Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.917 | 0.977 | 0.793 | 0.815 | 23 | 28 |  |  |

Answer quality, run 2026-11-12, `gpt-5.4-nano`, top 5 passages ([mini_wiki](../../../../results/answer_mini_wiki.md), [vi_legal](../../../../results/answer_vi_legal.md)).

mini_wiki (150 questions, English):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.797 | 97.3% | 8.7% | 30 |  | 900 | 633 | 757 | 0 | 0 |

vi_legal (100 questions, Vietnamese legal):

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.620 | 98.0% | 18.0% | 34 |  | 1224 | 1200 | 1153 | 0 | 0 |

Answer quality is judged by the same model that writes the answers (`gpt-5.4-nano`), which can flatter its own answers. All methods share the generator and the judge. Abstentions count as incorrect and faithful, so a method that abstains more can score higher on faithfulness. Correctness and faithfulness are averaged over the questions the judge scored, so the denominators can differ when a judge call fails (the `Judge failed` column). `Passage tokens (est.)` is estimated as characters / 3. The `vi_RAG` gold answers were aligned with LLMs per the dataset card (Mixtral 8x22B, GPT-4). The rag-mini-wikipedia data is `cc-by-3.0` per its dataset card.

The answer eval was run twice on 2026-11-12; the first run had a faithfulness-judge bug, since fixed, and the tables here are from the second run. Correctness moved by up to 0.07 between the two runs with no change to the correctness judge, so differences smaller than that are within run-to-run variance and should not be read as one method beating another.

This method is the fixed-size baseline in the chunking comparison, see the [`parse_and_chunk` results](../parse_and_chunk/README.md#chunking).

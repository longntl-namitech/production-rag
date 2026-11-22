<div align="center">

# LLM rerank

**Retrieve 20 candidates, have an LLM score each one, keep the best 5.**

</div>

<p align="center"><img src="../../../../docs/diagrams/methods/llm_rerank.svg" alt="Dense retrieval returns 20 candidate passages, an LLM scores each passage from 0 to 10, and the passages are sorted by score and cut to the top 5." width="100%"></p>

| | |
|---|---|
| **Stage** | Rerank |
| **Extra** | `local-models` |
| **LLM** | at query time (`OPENAI_API_KEY`) |
| **Eval variants** | retrieval eval `llm_rerank` (`core/eval/run_llm.py`), next to `naive_rag` |
| **Compare with** | [`rerank`](../rerank/README.md): that one scores with a local cross-encoder, this one asks an LLM |
| **Headline** | to be filled from the eval run |

## When to use
- A cheap first-stage search puts the right passage in the top 20 but not in the top 5, and one LLM call per query is affordable.

## When not to
- Latency or cost per query matters more than ranking quality: a local cross-encoder (`rerank`) adds no LLM call.
- The first stage already ranks the right passage near the top.

## How it works
- [`rerank.py`](rerank.py): one LLM call gets the question and every candidate passage, numbered, and replies with a score from 0 to 10 per passage (JSON, capped at 512 tokens). The passages are sorted by score; two passages with the same score keep their search order.
- The score is the model's own judgement, not a calibrated probability. It is only used to sort.
- Fallback: a reply that cannot be used (not JSON, a passage missing or repeated, an id or score out of range, a reply cut at the cap, a failed call) returns the passages in their search order. `BudgetExceeded` propagates.
- [`pipeline.py`](pipeline.py): candidates are the top 20 of [`naive_rag`](../naive_rag/README.md) (fixed 800-character chunks, dense), then `rerank`, then the first `k`.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.llm_rerank.example
```

Without `OPENAI_API_KEY` the example prints the search results and stops.

## Results

To be filled from the eval run.

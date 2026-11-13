# Step-back: search the question and a more general one

Extra: `local-models`. Needs an LLM at query time (`OPENAI_API_KEY`).

Differs from [`multi_query`](../multi_query/README.md): one deliberately more abstract question, not several rephrasings at the same level of detail.

## When to use
- A specific question needs background that a more general passage holds, for example "why did the 1929 crash hit German banks" and a passage on the causes of the Great Depression.

## When not to
- The question is already general, or the answer sits in one specific passage: the extra query only adds noise to the merge.

## How it works
- [`generate.py`](generate.py): the LLM writes one more general question (JSON, capped at 128 tokens). Any bad output, or a step-back that repeats the question, gives the question alone, so search still runs. `BudgetExceeded` propagates.
- [`pipeline.py`](pipeline.py): `search` is the `multi_query` search. The question and the step-back question each retrieve on their own, then each chunk is kept once with its best score. The default base is `hybrid_weighted_rrf`.

Source: Zheng et al., [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117) (2023). Own prompt, reimplemented from the paper.

## Run

```bash
OPENAI_API_KEY=... uv run --extra local-models python -m production_rag.methods.step_back.example
```

Without a key the example searches the question and says so.

## Results
Not measured yet.

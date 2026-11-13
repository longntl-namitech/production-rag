# Query decomposition: one search per part of the question

Extra: `local-models`. Needs an LLM at query time (`OPENAI_API_KEY`).

Differs from [`multi_query`](../multi_query/README.md): the sub-questions cover different parts of the question, not the same question reworded.

## When to use
- A question asks for several things at once ("who wrote it and when was it built"), and no single passage answers all of them.

## When not to
- A question with one part: the LLM returns it as the only sub-question and the search is the same as plain search, plus one LLM call.

## How it works
- [`generate.py`](generate.py): the LLM splits the question into at most 3 sub-questions (JSON, capped at 256 tokens; the cap is the `n` argument). Duplicates are dropped. Any bad output or an empty list gives the original question, so search still runs. `BudgetExceeded` propagates.
- [`pipeline.py`](pipeline.py): `search` is the `multi_query` search. Each sub-question retrieves on its own, then each chunk is kept once with its best score. With a reranking base, pass `rerank_query=<the original question>`.

Source: Khot et al., [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406) (2022). Own prompt, reimplemented from the paper.

## Run

```bash
OPENAI_API_KEY=... uv run --extra local-models python -m production_rag.methods.query_decomposition.example
```

Without a key the example searches the question as is and says so.

## Results
Not measured yet.

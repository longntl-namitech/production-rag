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
Pending: filled in from results/ after the eval runs.

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
No public numbers. The eval datasets index each document as one passage, so there are no neighbours to add. Behaviour is covered by tests (`test_expand_prev_next`, `test_expand_full_document`, `test_context_expansion_applied`).

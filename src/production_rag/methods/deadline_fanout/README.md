# Deadline budget and spec fan-out

Extra: `local-models`

## When to use
- One request needs several searches with different filters or `k` (for example FAQ entries and documents), and a caller with a hard latency budget.

## How it works
- The `deadline_ms` argument of `search` (default 4,000) creates one budget for the whole call ([`pipeline/deadline.py`](../../core/pipeline/deadline.py)).
- The query is embedded once, then each spec runs as its own task under the remaining budget ([`pipeline/search.py`](../../core/pipeline/search.py)).
- A spec that raises or runs out of time comes back with `status="failed"` and `error` set to the exception type; the other specs still return. Only the exception type is returned, never its message.
- If the budget is spent before the query is embedded, `search` raises `DeadlineExpired` (from `production_rag.core.pipeline.search`) instead of returning failed specs.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.deadline_fanout.example
```

## Lessons from building this repo
- Warm every model before the first timed query (`warm_up`). The first real request spent its whole budget loading the dense model and reranker and failed.

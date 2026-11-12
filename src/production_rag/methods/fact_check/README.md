# Fact check

Extra: `local-models` for the default hybrid evidence search (`base="bm25"` works without it)

LLM: `OPENAI_API_KEY` (model `OPENAI_MODEL`, default `gpt-5.4-nano`)

## When to use
- A statement (an answer, a message, a paragraph) must be checked against a document collection, and it may mix several facts.

## How it works
- `extract_claims` asks the model for up to 5 self-contained claims, each with a short search query. Opinions, questions and greetings are skipped. Blank entries and duplicates are dropped.
- `check_claim` retrieves evidence for each claim with any variant (`base`, default `hybrid_weighted_rrf`) and `judge_claim` labels it `supported`, `refuted` or `unverified` from the retrieved passages only. No evidence means `unverified` with no LLM call.
- A verdict the model cannot give cleanly (invalid JSON, a label outside the three, a reply cut at the token cap, a failed call) has no label and an error. It is never scored as `unverified`. A 401 or 403 raises `AuthFailed`, and `BudgetExceeded` propagates.
- `aggregate` gives the statement label: any `refuted` claim makes it `refuted`, else any `unverified` claim makes it `unverified`, else `supported`. A statement with no checkable claim is `not_relevant`. A claim with no label counts as `unverified` here and keeps its error on its `ClaimResult`.

## Production lessons
- Claims are extracted first, so a statement that mixes facts gets one verdict per claim.
- One refuted claim makes the whole statement refuted.
- In production a verdict below a confidence threshold was downgraded to `unverified`. That is not done here, because the chat API gives no calibrated probability.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.fact_check.example
```

Without `OPENAI_API_KEY` the example prints the evidence for two claims and stops.

## Results
Pending: the fact-check eval has not been run yet.

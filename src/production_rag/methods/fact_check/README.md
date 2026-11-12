# Fact check

Extra: `local-models` for the default hybrid evidence search (`base="bm25"` works without it)

LLM: `OPENAI_API_KEY` (model `OPENAI_MODEL`, default `gpt-5.4-nano`)

## When to use
- A statement (an answer, a message, a paragraph) must be checked against a document collection, and it may mix several facts.

## How it works
- `extract_claims` asks the model for up to 5 self-contained claims, each with a short search query. Opinions, questions, greetings and promises are skipped. Blank entries and duplicates are dropped.
- `check_claim` retrieves evidence for each claim with any variant (`base`, default `hybrid_weighted_rrf`) and `judge_claim` labels it `supported`, `refuted` or `unverified` from the retrieved passages only. No evidence means `unverified` with no LLM call.
- A verdict the model cannot give cleanly (invalid JSON, a label outside the three, a reply cut at the token cap, a failed call) has no label and an error. In the eval such a verdict is `failed` and is never scored as `unverified`. A 401 or 403 raises `AuthFailed`, and `BudgetExceeded` propagates.
- `aggregate` gives the statement label: any `refuted` claim makes it `refuted`, else any `unverified` claim makes it `unverified`, else `supported`. A statement with no checkable claim is `not_relevant`. In this statement label only, a failed claim counts as `unverified`, and it keeps its error on its `ClaimResult`.

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

300 SciFact dev claims against 5,183 abstracts, `gpt-5.4-nano`, base `hybrid_weighted_rrf`, k = 5, run on 2026-11-12. The eval gives verdicts only: each claim is judged as given, with no claim extraction. No model acts as a judge, because the labels are gold. Full tables: [`results/fact_check_scifact.md`](../../../../results/fact_check_scifact.md). The tables below are from run 1.

The dev split has 124 supported, 64 refuted and 112 no-evidence claims. Failed items are excluded from accuracy, precision and recall.

| Label | Gold | Predicted | Correct | Precision | Recall |
|---|---|---|---|---|---|
| supported | 124 | 118 | 93 | 0.788 | 0.750 |
| refuted | 64 | 39 | 24 | 0.615 | 0.375 |
| unverified | 112 | 143 | 94 | 0.657 | 0.839 |

Accuracy 0.703 on 300 claims, 0 failed, unverified 47.7%. On the 188 supported or refuted claims, the gold abstract is in the top 5 for 180 (95.7%); accuracy 0.639 when found (115/180), 0.250 when not (2/8).

| Gold \ predicted | supported | refuted | unverified | failed |
|---|---|---|---|---|
| supported | 93 | 7 | 24 | 0 |
| refuted | 15 | 24 | 25 | 0 |
| unverified | 10 | 8 | 94 | 0 |

### Run-to-run

Two full runs gave accuracy 0.703 and 0.740, with the same label on 89.3% of the 300 claims ([`results/fact_check_scifact_runs.md`](../../../../results/fact_check_scifact_runs.md)). Differences inside that spread are not evidence.

| Run | Date | Model | Base | k | Claims | Accuracy |
|---|---|---|---|---|---|---|
| A | 2026-11-12 | gpt-5.4-nano | hybrid_weighted_rrf | 5 | 300 | 0.703 |
| B | 2026-11-12 | gpt-5.4-nano | hybrid_weighted_rrf | 5 | 300 | 0.740 |

### Notes

- Refuted recall is the weakest (0.375 in run 1): 25 of the 64 refuted claims got `unverified`.
- SciFact's NOT ENOUGH INFO label is scored as `unverified`.
- Evidence comes from open retrieval over 5,183 abstracts, so a claim can miss its gold abstract. On the 188 supported or refuted claims (not the 112 unverified ones), the gold abstract is in the top 5 for 180 (95.7%); accuracy is 0.639 when it is found (115/180) and 0.250 when not (2/8).
- `not_relevant` cannot occur on single claims, so it is not scored.
- SciFact claims are CC BY-NC 2.0 (non-commercial) per the `allenai/scifact` card. They are downloaded at runtime from allenai's release tarball and not stored in the repo.

# Hybrid fusion: RRF, DBSF, weighted RRF

Extra: `local-models`

## When to use
- You do not know in advance whether queries will be keyword-like or paraphrased, and want one method that is never the worst.

## When not to
- One leg is clearly stronger on your data. On TVPL, dense alone beats every hybrid here: fusing in a weaker BM25 leg pulls good dense results down.

## How it works
- [`fusion.py`](../../core/fusion.py): each leg is a Qdrant prefetch (`limit = k * 4`) and the outer query fuses them inside Qdrant, so there is one round trip.
  - `hybrid_rrf`: reciprocal rank fusion, uses ranks only.
  - `hybrid_dbsf`: distribution-based score fusion, normalises each leg's scores before adding them.
  - `hybrid_weighted_rrf`: RRF with weights per leg, dense 1.0 and sparse 0.5 (Qdrant 1.17+).
- Several subqueries are merged by taking each chunk's best score ([multi-query](../multi_query/README.md)).

## Run

```bash
uv run --extra local-models python -m production_rag.methods.hybrid_fusion.example
```

## Results
SciFact:

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 21 | 26 |  |  |
| `bm25` | 0.809 | 0.921 | 0.647 | 0.683 | 7 | 8 |  |  |
| `hybrid_rrf` | 0.882 | 0.973 | 0.718 | 0.755 | 23 | 28 |  |  |
| `hybrid_dbsf` | 0.880 | 0.957 | 0.716 | 0.752 | 23 | 27 |  |  |
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.736 | 0.769 | 23 | 26 |  |  |

TVPL (Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |
| `bm25` | 0.800 | 0.915 | 0.641 | 0.669 | 7 | 9 |  |  |
| `hybrid_rrf` | 0.912 | 0.974 | 0.757 | 0.786 | 26 | 36 |  |  |
| `hybrid_dbsf` | 0.899 | 0.963 | 0.773 | 0.793 | 25 | 32 |  |  |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 26 | 31 |  |  |

- Weighted RRF is the best hybrid on both sets and the best method overall on SciFact.
- Plain RRF treats the two legs as equals; when the sparse leg is clearly weaker (TVPL) it costs 0.03 nDCG@10 against dense alone.
- The weights were set once (dense 1.0, sparse 0.5) and not tuned on these test sets.

## Lessons from production
- Measured on internal data, data not published. The same audit found that scores from three sources sat on incompatible scales, and that chunks of one document collapsed into the worst-ranked one.
- Measured on internal data, data not published. Hybrid beat dense alone by a little: recall@10 0.416 vs 0.403, keyword alone 0.301.

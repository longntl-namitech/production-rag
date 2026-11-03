# BM25: IDF and Vietnamese stopwords

Extra: none for search; the eval and example index dense vectors too, so they need `local-models`

Three variants share one index: `bm25_no_idf`, `bm25` (IDF on) and `bm25_vi_stopwords` (Vietnamese stopwords removed before encoding).

## When to use
- Exact terms carry the meaning: codes, article numbers, product names, rare words.
- As the second leg of a hybrid search, to catch what the dense model blurs.

## When not to
- Alone, on paraphrased questions: it trails dense retrieval by 0.08 to 0.15 nDCG@10 on both datasets here.

## How it works
- [`sparse/bm25.py`](../../core/sparse/bm25.py): fastembed `Qdrant/bm25` produces term-frequency weights; Qdrant applies IDF at query time when the sparse vector is created with `modifier=IDF` ([`store.py`](../../core/store.py)).
- The same weights are also stored in `sparse_noidf`, a sparse vector with no modifier, to measure what IDF is worth.
- The Vietnamese variant removes stopword phrases from [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords) on whole-syllable boundaries, longest phrase first, because many entries span several syllables ("bởi vì").

## Run

```bash
uv run --extra local-models python -m production_rag.methods.bm25.example
```

## Results
SciFact:

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `bm25_no_idf` | 0.714 | 0.859 | 0.572 | 0.601 | 6 | 8 |  |  |
| `bm25` | 0.809 | 0.921 | 0.647 | 0.683 | 7 | 8 |  |  |
| `bm25_vi_stopwords` | 0.793 | 0.886 | 0.627 | 0.661 | 7 | 8 |  |  |

TVPL (Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `bm25_no_idf` | 0.751 | 0.873 | 0.613 | 0.636 | 8 | 9 |  |  |
| `bm25` | 0.800 | 0.915 | 0.641 | 0.669 | 7 | 9 |  |  |
| `bm25_vi_stopwords` | 0.768 | 0.910 | 0.617 | 0.645 | 7 | 11 |  |  |

- IDF is worth 0.08 nDCG@10 on SciFact and 0.03 on TVPL.
- Removing Vietnamese stopwords made BM25 worse on TVPL (0.669 to 0.645 nDCG@10). The general-purpose list removes words that carry meaning in legal questions. For example "Người lao động được gia hạn trả nợ vay vốn để đi làm việc ở nước ngoài trong trường hợp nào?" becomes "lao động gia hạn nợ vay vốn đi trường hợp": "người", "trả", "làm việc" and "nước ngoài" are gone. IDF already down-weights words that are frequent in the corpus, which is what a stopword list tries to do by hand.

## Lessons from production
- Measured on internal data, data not published. An audit of a production hybrid index found the BM25 leg had no IDF, so Vietnamese stopwords drove its ranking.

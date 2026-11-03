# production-rag

"Production" here means the methods come from production use, from naive to advanced RAG. Each one is written up next to runnable code and measured on public data. This is a lab to read and try methods in, not a deployable service.

## Results

nDCG@10 and the other metrics are averaged over the test queries; latency is in-process on CPU (Apple silicon, one query at a time). Full tables: [SciFact](results/scifact.md), [TVPL](results/tvpl.md).

**SciFact** (English scientific claims, 300 queries, 5,183 documents)

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 21 | 26 |  |  |
| `bm25_no_idf` | 0.714 | 0.859 | 0.572 | 0.601 | 6 | 8 |  |  |
| `bm25` | 0.809 | 0.921 | 0.647 | 0.683 | 7 | 8 |  |  |
| `bm25_vi_stopwords` | 0.793 | 0.886 | 0.627 | 0.661 | 7 | 8 |  |  |
| `hybrid_rrf` | 0.882 | 0.973 | 0.718 | 0.755 | 23 | 28 |  |  |
| `hybrid_dbsf` | 0.880 | 0.957 | 0.716 | 0.752 | 23 | 27 |  |  |
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.736 | 0.769 | 23 | 26 |  |  |
| `hybrid_rrf_rerank` | 0.752 | 0.973 | 0.636 | 0.660 | 985 | 1096 | 6.0% | 11.0% |

**TVPL** (Vietnamese legal questions, 1,000 sampled queries, 10,576 documents)

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |
| `bm25_no_idf` | 0.751 | 0.873 | 0.613 | 0.636 | 8 | 9 |  |  |
| `bm25` | 0.800 | 0.915 | 0.641 | 0.669 | 7 | 9 |  |  |
| `bm25_vi_stopwords` | 0.768 | 0.910 | 0.617 | 0.645 | 7 | 11 |  |  |
| `hybrid_rrf` | 0.912 | 0.974 | 0.757 | 0.786 | 26 | 36 |  |  |
| `hybrid_dbsf` | 0.899 | 0.963 | 0.773 | 0.793 | 25 | 32 |  |  |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 26 | 31 |  |  |
| `hybrid_rrf_rerank` | 0.906 | 0.974 | 0.792 | 0.810 | 812 | 1013 | 11.0% | 4.9% |

What stands out:
- IDF matters: BM25 without it loses 0.03 to 0.08 nDCG@10 ([BM25](src/production_rag/methods/bm25/README.md)).
- A general Vietnamese stopword list made BM25 worse on legal questions, because it removes words that carry meaning there ([BM25](src/production_rag/methods/bm25/README.md)).
- Weighted RRF (dense 1.0, sparse 0.5) is the best hybrid on both sets; on TVPL dense alone is still slightly better ([hybrid fusion](src/production_rag/methods/hybrid_fusion/README.md)).
- The multilingual cross-encoder helps on Vietnamese legal questions and hurts on English scientific claims, at about 1 s per query on CPU ([rerank](src/production_rag/methods/rerank/README.md)).

## Methods

| Folder | Variants (eval names) | Extra | What it does |
|---|---|---|---|
| [`dense`](src/production_rag/methods/dense/README.md) | `dense` | `local-models` | Jina v5 nano embeddings |
| [`bm25`](src/production_rag/methods/bm25/README.md) | `bm25_no_idf`, `bm25`, `bm25_vi_stopwords` | none for search | BM25 without IDF, with IDF, and with Vietnamese stopwords removed first |
| [`hybrid_fusion`](src/production_rag/methods/hybrid_fusion/README.md) | `hybrid_rrf`, `hybrid_dbsf`, `hybrid_weighted_rrf` | `local-models` | Dense + BM25 fused in Qdrant: reciprocal rank fusion, distribution-based score fusion, RRF with dense weighted 2:1 |
| [`rerank`](src/production_rag/methods/rerank/README.md) | `hybrid_rrf_rerank` | `local-models` | Hybrid RRF, then a cross-encoder over the top candidates |
| [`deadline_fanout`](src/production_rag/methods/deadline_fanout/README.md) | none | `local-models` | Deadline budget and spec fan-out; no public numbers yet |
| [`multi_query`](src/production_rag/methods/multi_query/README.md) | none | `local-models` | Multi-query merge; no public numbers yet |
| [`context_expansion`](src/production_rag/methods/context_expansion/README.md) | none | `local-models` | Neighbour chunks or the full document around each hit; no public numbers yet |

## Quickstart

```bash
docker compose up -d qdrant
uv sync --extra local-models --extra eval

# index SciFact and score two methods (downloads the dataset)
uv run python -m production_rag.core.eval.run --dataset scifact --methods bm25 hybrid_fusion --query-limit 20 --name smoke

# run one method's example
uv run python -m production_rag.methods.hybrid_fusion.example
```

As a library:

```python
from production_rag.methods.hybrid_fusion.pipeline import search
```

## Adding a method

1. Add a folder under `src/production_rag/methods/` with `README.md`, `pipeline.py` and `example.py`. `pipeline.py` exports `VARIANTS`, `DEFAULT` and `search`.
2. Add the name to `METHOD_NAMES` in `src/production_rag/methods/__init__.py`.
3. A new vector kind or backend goes in `core/`.
4. Run the eval on both datasets, then `python -m production_rag.core.eval.report`.

## Tests

```bash
uv run pytest                              # no network, no Qdrant, no extras
uv run --extra local-models pytest -m integration
```

## Roadmap

Next: `naive_rag`, `parse_and_chunk`, `fact_check`. After that: query rewrite and speculative search, LLM rerank, cache, semantic chunking, embedding comparison, `cpu_embedding_batching`, testset generation. Known gaps: [`BACKLOG.md`](BACKLOG.md).

## Models, data and licenses

- Dense: [`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval), CC-BY-NC-4.0 (non-commercial). Swap it through the [`Embedder`](src/production_rag/core/embedders/base.py) protocol or point `EMBEDDING_BASE_URL` at an OpenAI-compatible `/embeddings` endpoint.
- Sparse: fastembed `Qdrant/bm25`. Rerank: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).
- Vietnamese stopwords: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords), MIT.
- Datasets are downloaded at runtime and not stored here: [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) and [GreenNode/TVPL-Retrieval-VN](https://huggingface.co/datasets/GreenNode/TVPL-Retrieval-VN), both CC-BY-SA-4.0.

Code: MIT.

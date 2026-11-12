# production-rag

"Production" here means the methods come from production use, from naive to advanced RAG. Each one is written up next to runnable code and measured on public data. This is a lab to read and try methods in, not a deployable service.

## Results

nDCG@10 and the other metrics are averaged over the test queries; latency is in-process on CPU (Apple silicon, one query at a time). Full tables: [SciFact](results/scifact.md), [TVPL](results/tvpl.md), and for `naive_rag` [SciFact](results/scifact_naive_rag.md), [TVPL](results/tvpl_naive_rag.md).

**SciFact** (English scientific claims, 300 queries, 5,183 documents)

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.875 | 0.962 | 0.715 | 0.751 | 22 | 26 |  |  |
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
| `naive_rag` | 0.917 | 0.977 | 0.793 | 0.815 | 23 | 28 |  |  |
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |
| `bm25_no_idf` | 0.751 | 0.873 | 0.613 | 0.636 | 8 | 9 |  |  |
| `bm25` | 0.800 | 0.915 | 0.641 | 0.669 | 7 | 9 |  |  |
| `bm25_vi_stopwords` | 0.768 | 0.910 | 0.617 | 0.645 | 7 | 11 |  |  |
| `hybrid_rrf` | 0.912 | 0.974 | 0.757 | 0.786 | 26 | 36 |  |  |
| `hybrid_dbsf` | 0.899 | 0.963 | 0.773 | 0.793 | 25 | 32 |  |  |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 26 | 31 |  |  |
| `hybrid_rrf_rerank` | 0.906 | 0.974 | 0.792 | 0.810 | 812 | 1013 | 11.0% | 4.9% |

`naive_rag` searches 800-character chunks (a document counts once, at its best chunk) and was run on 2026-11-11; the other rows are from the first release and were not rerun.

### Answer quality

Retrieval is scored against labelled documents; this section scores the generated answer. Each method retrieves the top 5 passages (k = 5), the same generator (`gpt-5.4-nano`) answers from them, and an LLM judge scores correctness against the gold answer and faithfulness to the passages. Judged by the same model that answers, which can flatter its own answers; all methods share the generator and judge. Run 2026-11-12. The rag-mini-wikipedia data is `cc-by-3.0` per its dataset card. Full tables: [mini_wiki](results/answer_mini_wiki.md), [vi_legal](results/answer_vi_legal.md).

**rag-mini-wikipedia** (English, 150 sampled questions, 3200 documents)

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.797 | 97.3% | 8.7% | 30 |  | 900 | 633 | 757 | 0 | 0 |
| `hybrid_fusion` | 0.813 | 96.7% | 8.7% | 31 |  | 837 | 655 | 793 | 0 | 0 |
| `rerank` | 0.823 | 96.7% | 8.7% | 137 |  | 821 | 617 | 735 | 0 | 0 |
| `multi_query` | 0.817 | 96.0% | 8.0% | 41 | 1128 | 804 | 834 | 801 | 0 | 0 |
| `context_expansion` | 0.830 | 96.0% | 8.7% | 35 |  | 804 | 744 | 932 | 0 | 0 |

**vi_RAG LegalRAG** (Vietnamese legal, 100 sampled questions, 2125 documents)

| Method | Correctness | Faithfulness | Abstained | Retrieval p50 ms | Rewrite p50 ms | Generation p50 ms | RAG tokens / question | Passage tokens (est.) | Failed | Judge failed |
|---|---|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.620 | 98.0% | 18.0% | 34 |  | 1224 | 1200 | 1153 | 0 | 0 |
| `hybrid_fusion` | 0.580 | 95.0% | 23.0% | 38 |  | 1260 | 1213 | 1169 | 0 | 0 |
| `rerank` | 0.630 | 96.0% | 15.0% | 156 |  | 1154 | 1193 | 1136 | 0 | 0 |
| `multi_query` | 0.610 | 97.0% | 18.0% | 66 | 1372 | 1190 | 1436 | 1171 | 0 | 0 |
| `context_expansion` | 0.685 | 92.0% | 15.0% | 37 |  | 1209 | 2455 | 2570 | 0 | 0 |

Abstentions count as incorrect and faithful, so faithfulness is higher for a method that abstains more. Correctness and faithfulness are averaged over judged questions, so the denominators can differ when a judge call fails (no judge failures in these runs). Passage tokens are estimated as characters / 3. The `vi_RAG` gold answers were aligned with LLMs per the dataset card, so the Vietnamese scores are not a human-labelled measure. The rag-mini-wikipedia data is `cc-by-3.0` per its dataset card.

The answer eval was run twice on 2026-11-12; the first run had a faithfulness-judge bug, since fixed, and the tables here are from the second run. Correctness moved by up to 0.07 between the two runs with no change to the correctness judge, so differences smaller than that are within run-to-run variance and should not be read as one method beating another.

### Chunking

Retrieval only, dense only, run on 2026-11-12, so chunking is the only difference. The `naive_rag` row is fixed 800-character chunks; the other rows in the tables above are untouched. A document counts once, at its best chunk, so recall@100 counts the distinct documents in the top 100 chunks and smaller chunks are penalised by construction. A default `eval.run` (no `--methods`) also builds the three chunk-variant collections. Full tables: [SciFact](results/scifact_chunking.md), [TVPL](results/tvpl_chunking.md). Details: [parse and chunk](src/production_rag/methods/parse_and_chunk/README.md).

**SciFact**

| Collection | Chunker | Size | Overlap | Chunks | Mean chars |
|---|---|---|---|---|---|
| `eval-scifact-c800` | fixed | 800 | 0 | 11805 | 615 |
| `eval-scifact-r800` | recursive | 800 | 0 | 12315 | 589 |
| `eval-scifact-r800o200` | recursive | 800 | 200 | 16143 | 583 |
| `eval-scifact-r400` | recursive | 400 | 0 | 24724 | 293 |

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.875 | 0.962 | 0.715 | 0.751 | 20 | 27 |  |  |
| `chunk_recursive_800` | 0.878 | 0.958 | 0.715 | 0.750 | 22 | 27 |  |  |
| `chunk_recursive_800_o200` | 0.870 | 0.948 | 0.717 | 0.750 | 22 | 28 |  |  |
| `chunk_recursive_400` | 0.871 | 0.947 | 0.706 | 0.743 | 21 | 27 |  |  |

**TVPL**

| Collection | Chunker | Size | Overlap | Chunks | Mean chars |
|---|---|---|---|---|---|
| `eval-tvpl-c800` | fixed | 800 | 0 | 17726 | 559 |
| `eval-tvpl-r800` | recursive | 800 | 0 | 18556 | 534 |
| `eval-tvpl-r800o200` | recursive | 800 | 200 | 23813 | 525 |
| `eval-tvpl-r400` | recursive | 400 | 0 | 35277 | 281 |

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `naive_rag` | 0.917 | 0.977 | 0.793 | 0.815 | 20 | 28 |  |  |
| `chunk_recursive_800` | 0.914 | 0.976 | 0.787 | 0.810 | 22 | 28 |  |  |
| `chunk_recursive_800_o200` | 0.909 | 0.970 | 0.781 | 0.802 | 22 | 26 |  |  |
| `chunk_recursive_400` | 0.905 | 0.969 | 0.770 | 0.794 | 22 | 26 |  |  |

### Parsing quality

Token recall and precision on 15 public documents, run 2026-11-12 ([full table](results/parse_quality.md)). PDF is scored against py-pdf/benchmarks ground truth; the docx, pptx, xlsx and html references are docling's own output, so those scores are agreement with another converter. Order and layout are not scored.

| Format | Files | Failed | Token recall | Token precision | p50 ms |
|---|---|---|---|---|---|
| pdf | 7 | 0 | 0.982 | 0.920 | 502 |
| docx | 4 | 0 | 0.989 | 1.000 | 4 |
| pptx | 1 | 0 | 0.848 | 0.989 | 134 |
| xlsx | 1 | 0 | 0.629 | 0.993 | 159 |
| html | 2 | 0 | 1.000 | 0.982 | 28 |
| all | 15 | 0 | 0.954 | 0.959 | 235 |

The `all` row is a mean over files, so the 7 PDFs weigh more than the other formats.

### Fact check

300 SciFact dev claims against 5,183 abstracts, `gpt-5.4-nano`, base `hybrid_weighted_rrf`, k = 5, run 2026-11-12. Verdicts only, scored against the gold labels (no LLM judge). Two full runs gave accuracy 0.703 and 0.740, with the same label on 89.3% of claims; the table is from run 1 and differences inside that spread are not evidence. Details: [fact check](src/production_rag/methods/fact_check/README.md), [results](results/fact_check_scifact.md), [runs](results/fact_check_scifact_runs.md).

| Label | Gold | Predicted | Correct | Precision | Recall |
|---|---|---|---|---|---|
| supported | 124 | 118 | 93 | 0.788 | 0.750 |
| refuted | 64 | 39 | 24 | 0.615 | 0.375 |
| unverified | 112 | 143 | 94 | 0.657 | 0.839 |

Accuracy 0.703 on 300 claims, 0 failed, unverified 47.7%. On the 188 supported or refuted claims, the gold abstract is in the top 5 for 180 (95.7%); accuracy 0.639 when found (115/180), 0.250 when not (2/8).

What stands out:
- IDF matters: BM25 without it loses 0.03 to 0.08 nDCG@10 ([BM25](src/production_rag/methods/bm25/README.md)).
- A general Vietnamese stopword list made BM25 worse on legal questions, because it removes words that carry meaning there ([BM25](src/production_rag/methods/bm25/README.md)).
- Weighted RRF (dense 1.0, sparse 0.5) is the best hybrid on both sets; on TVPL dense alone is still slightly better ([hybrid fusion](src/production_rag/methods/hybrid_fusion/README.md)).
- The multilingual cross-encoder helps on Vietnamese legal questions and hurts on English scientific claims, at about 1 s per query on CPU ([rerank](src/production_rag/methods/rerank/README.md)).
- Chunked `naive_rag` retrieval is close to `dense` on whole documents: nDCG@10 0.751 vs 0.762 on SciFact and 0.815 vs 0.818 on TVPL ([naive RAG](src/production_rag/methods/naive_rag/README.md)).
- On answers, correctness spans 0.03 across the methods on mini_wiki (0.797 to 0.830), inside the 0.07 run-to-run variance, so no method is shown to be better there. On vi_legal it spans 0.105 (0.580 to 0.685), and the one gap larger than 0.07 is `context_expansion` against `hybrid_fusion`. Faithfulness is 96.0% to 97.3% on mini_wiki and 92.0% to 98.0% on vi_legal; abstentions are 8.0% to 8.7% on mini_wiki and 15.0% to 23.0% on vi_legal. These are single runs on small samples.
- On vi_legal, `context_expansion` sends about twice the passage tokens of the others (2570 vs 1136 to 1171) and has the highest correctness (0.685) and the lowest faithfulness (92.0%). Its correctness gap to `naive_rag` (0.685 vs 0.620) is 0.065, within the run-to-run variance ([context expansion](src/production_rag/methods/context_expansion/README.md)).
- `multi_query` adds about 1.2 to 1.5 s of rewrite time per question ([multi query](src/production_rag/methods/multi_query/README.md)).
- Fan-out under a 150 ms deadline skips the rerank spec on every query and answers in 22 ms p50, against 366 ms at 2000 ms ([deadline fan-out](src/production_rag/methods/deadline_fanout/README.md)).
- Chunking: fixed 800 is best or tied on nDCG@10 on both sets, a 200-character overlap does not help, and recursive 400 is lowest ([parse and chunk](src/production_rag/methods/parse_and_chunk/README.md)).
- Parsing: PDF token recall is 0.982 and precision 0.920; the xlsx (0.629) and pptx (0.848) recall gaps come from docling's reference output, not from missing cells ([parse and chunk](src/production_rag/methods/parse_and_chunk/README.md)).
- Fact check: accuracy is 0.703 and 0.740 over two runs, and refuted recall is the weakest (0.375 in run 1): 25 of 64 refuted claims got `unverified` ([fact check](src/production_rag/methods/fact_check/README.md)).

## Methods

| Folder | Variants (eval names) | Extra | What it does |
|---|---|---|---|
| [`naive_rag`](src/production_rag/methods/naive_rag/README.md) | `naive_rag` | `local-models` + `OPENAI_API_KEY` | Fixed 800-char chunks, dense top 5, one prompt, generate |
| [`dense`](src/production_rag/methods/dense/README.md) | `dense` | `local-models` | Jina v5 nano embeddings |
| [`bm25`](src/production_rag/methods/bm25/README.md) | `bm25_no_idf`, `bm25`, `bm25_vi_stopwords` | none for search | BM25 without IDF, with IDF, and with Vietnamese stopwords removed first |
| [`hybrid_fusion`](src/production_rag/methods/hybrid_fusion/README.md) | `hybrid_rrf`, `hybrid_dbsf`, `hybrid_weighted_rrf` | `local-models` | Dense + BM25 fused in Qdrant: reciprocal rank fusion, distribution-based score fusion, RRF with dense weighted 2:1 |
| [`rerank`](src/production_rag/methods/rerank/README.md) | `hybrid_rrf_rerank` | `local-models` | Hybrid RRF, then a cross-encoder over the top candidates |
| [`deadline_fanout`](src/production_rag/methods/deadline_fanout/README.md) | none | `local-models` | Deadline budget and spec fan-out; latency bench in its README |
| [`multi_query`](src/production_rag/methods/multi_query/README.md) | none | `local-models` | Multi-query merge; answer numbers in its README |
| [`context_expansion`](src/production_rag/methods/context_expansion/README.md) | none | `local-models` | Neighbour chunks or the full document around each hit; answer numbers in its README |
| [`parse_and_chunk`](src/production_rag/methods/parse_and_chunk/README.md) | `chunk_recursive_800`, `chunk_recursive_800_o200`, `chunk_recursive_400` | `parse` + `local-models` | PDF, DOCX, PPTX, XLSX, HTML, Markdown to Markdown with pip-only parsers; fixed, recursive and heading-aware chunking |
| [`fact_check`](src/production_rag/methods/fact_check/README.md) | none | `local-models` + `OPENAI_API_KEY` | claims from a statement, evidence per claim, verdict: supported, refuted, unverified; `not_relevant` when nothing is checkable |

## Quickstart

```bash
docker compose up -d qdrant
uv sync --extra local-models --extra eval

# index SciFact and score two methods (downloads the dataset)
uv run python -m production_rag.core.eval.run --dataset scifact --methods bm25 hybrid_fusion --query-limit 20 --name smoke

# run one method's example
uv run python -m production_rag.methods.hybrid_fusion.example

# answer eval (calls an LLM): estimate first, then run with a token cap
export OPENAI_API_KEY=...
uv run python -m production_rag.core.eval.answer --dataset mini_wiki --sample 20 --estimate-only
uv run python -m production_rag.core.eval.answer --dataset mini_wiki --sample 20 --max-tokens 400000

# parsing quality on public PDF and Office files (no LLM)
uv run --extra parse python -m production_rag.methods.parse_and_chunk.parse_eval

# chunking comparison (retrieval only, downloads the dataset)
uv run python -m production_rag.core.eval.run --dataset scifact --methods naive_rag parse_and_chunk --name scifact_chunking
# the same for the TVPL table: add --dataset tvpl and --name tvpl_chunking

# fact check (calls an LLM): estimate first, then a small capped run
uv run --extra eval --extra local-models python -m production_rag.methods.fact_check.eval --sample 30 --estimate-only
uv run --extra eval --extra local-models python -m production_rag.methods.fact_check.eval --sample 30 --max-tokens 100000
```

Estimate the tokens before every LLM run and give every LLM run a cap.

As a library:

```python
import asyncio

from production_rag.core.config import Settings
from production_rag.core.deps import build_deps, warm_up
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.hybrid_fusion import pipeline as hybrid_fusion
from production_rag.methods.naive_rag import pipeline as naive_rag


async def main() -> None:
    deps = build_deps(Settings.from_env())
    await warm_up(deps)
    await naive_rag.index(deps, "docs", {"d1": (None, "...")})
    llm = OpenAIChat.from_env(budget=TokenBudget(5_000))
    out = await naive_rag.answer("...", "docs", deps, llm)
    print(out.text)
    # retrieval only
    hits = await hybrid_fusion.search("...", "docs", deps)
    print(hits.rows)


asyncio.run(main())
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

Next: query rewrite and speculative search, LLM rerank, cache, semantic chunking, embedding comparison, `cpu_embedding_batching`, testset generation, chunking compared on answer quality. Known gaps: [`BACKLOG.md`](BACKLOG.md).

## Models, data and licenses

- Dense: [`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval), CC-BY-NC-4.0 (non-commercial). Swap it through the [`Embedder`](src/production_rag/core/embedders/base.py) protocol or point `EMBEDDING_BASE_URL` at an OpenAI-compatible `/embeddings` endpoint.
- Sparse: fastembed `Qdrant/bm25`. Rerank: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).
- Vietnamese stopwords: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords), MIT.
- LLM for generation and judging: OpenAI, default `gpt-5.4-nano`, set with `OPENAI_MODEL`.
- Answer eval data: [rag-datasets/rag-mini-wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) (`cc-by-3.0` per its card, Wikipedia text) and the `LegalRAG` config of [DuyTa/vi_RAG](https://huggingface.co/datasets/DuyTa/vi_RAG) (`apache-2.0` per its card; the id now redirects to sailor2/Vietnamese_RAG; its answers were aligned with LLMs per the card). VLQA is not used because it is not released yet.
- Datasets are downloaded at runtime and not stored here: [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) and [GreenNode/TVPL-Retrieval-VN](https://huggingface.co/datasets/GreenNode/TVPL-Retrieval-VN), both CC-BY-SA-4.0.
- Parse extra: `pdfplumber`, `python-docx`, `python-pptx`, `openpyxl`, `markdownify` and `beautifulsoup4`, all MIT.
- Parsing eval data: 7 arXiv papers (CC BY 4.0, except `arxiv-1602.06541v2`, CC BY-SA 4.0) with the [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks) ground truth (BSD-3-Clause, pinned commit `18ee8c4`), and the [docling](https://github.com/docling-project/docling) test fixtures (MIT, pinned commit `a7760ab`). Downloaded at runtime, not stored here.
- Fact-check data: [SciFact](https://huggingface.co/datasets/allenai/scifact) claims, CC BY-NC 2.0 (non-commercial) per the card, downloaded at runtime from allenai's release tarball and not stored here.

Code: MIT.

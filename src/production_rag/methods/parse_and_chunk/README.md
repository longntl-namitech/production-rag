# Parse and chunk

Extra: `parse` for PDF, Office and HTML (Markdown needs none); `local-models` for dense indexing

## When to use
- Your documents are files (PDF, Word, slides, spreadsheets, HTML, Markdown) and not plain text.
- You want to compare how the text is split before changing the retriever.

## How it works
- [`parsers.py`](parsers.py) turns each file into Markdown, with a different library per format:
  - PDF: `pdfplumber`, text plus tables as Markdown tables. A PDF with no text layer is reported as `no_text`.
  - DOCX: `python-docx`, headings, lists and tables.
  - PPTX: `python-pptx`, one `##` section per slide.
  - XLSX: `openpyxl`, one `##` section per sheet, rows as a Markdown table.
  - HTML: `beautifulsoup4` and `markdownify`, scripts and styles dropped.
  - Markdown: read as is.
- [`chunking.py`](../../core/chunking.py) has three strategies:
  - `fixed`: whitespace-split words packed up to the size.
  - `recursive`: split on paragraphs, lines, sentences, then words, and pack the pieces.
  - `markdown`: split on headings and prefix each chunk with its heading path.
- `recursive` and `markdown` accept an overlap: the next chunk starts with the tail of the previous one. `fixed` does not (it raises an error).
- [`pipeline.py`](pipeline.py) `index_files` parses the files, chunks them and calls `index_chunks`, which stores a dense vector and BM25 vectors for every chunk.
- The eval variants are `chunk_recursive_800`, `chunk_recursive_800_o200` and `chunk_recursive_400`. Each searches its own collection, built with its own chunker, size and overlap.

## Formats left out
- `.doc`, `.ppt`, `.xls`: legacy Office binary formats, convert them with LibreOffice or similar first.
- `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`: images need OCR (for example Tesseract).
- `.csv`, `.txt`, `.odt`, `.ods`, `.odp`, `.epub`, `.rtf`: not in the first release.
- Inside the supported formats, images, charts and embedded objects are not extracted, and a scanned PDF gives no text.

## Run

```bash
uv run --extra local-models python -m production_rag.methods.parse_and_chunk.example
uv run --extra local-models --extra parse python -m production_rag.methods.parse_and_chunk.example
```

The first run parses only the Markdown sample and tells you to install the `parse` extra for the HTML one. Pass file paths as arguments to parse your own files. No LLM is called.

## Results

### Parsing quality

Run on 2026-11-12 on 15 public documents. Full tables: [`results/parse_quality.md`](../../../../results/parse_quality.md).

The metric is token recall and token precision: text is NFKC-normalised and casefolded, tokens are `\w+` matches, and both scores are multiset overlaps against a reference. Order and layout are not scored. The 7 arXiv PDFs are scored against the [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks) ground truth. The docx, pptx, xlsx and html references are docling's own output, so for those files the score is agreement with another converter, not with a human transcription.

| Format | Files | Failed | Token recall | Token precision | p50 ms |
|---|---|---|---|---|---|
| pdf | 7 | 0 | 0.982 | 0.920 | 502 |
| docx | 4 | 0 | 0.989 | 1.000 | 4 |
| pptx | 1 | 0 | 0.848 | 0.989 | 134 |
| xlsx | 1 | 0 | 0.629 | 0.993 | 159 |
| html | 2 | 0 | 1.000 | 0.982 | 28 |
| all | 15 | 0 | 0.954 | 0.959 | 235 |

The `all` row is a mean over files, so the 7 PDFs weigh more than the other formats.

| Document | Format | Licence | Token recall | Token precision | Parse ms |
|---|---|---|---|---|---|
| `arxiv-1601.03642v1` | pdf | CC BY 4.0 | 0.992 | 0.981 | 235 |
| `arxiv-2201.00021v3` | pdf | CC BY 4.0 | 0.968 | 0.903 | 548 |
| `arxiv-2201.00022v2` | pdf | CC BY 4.0 | 0.978 | 0.709 | 502 |
| `arxiv-2201.00029v1` | pdf | CC BY 4.0 | 0.990 | 0.989 | 265 |
| `arxiv-2201.00069v1` | pdf | CC BY 4.0 | 0.975 | 0.928 | 446 |
| `arxiv-2201.00214v1` | pdf | CC BY 4.0 | 0.988 | 0.977 | 11765 |
| `arxiv-1602.06541v2` | pdf | CC BY-SA 4.0 | 0.979 | 0.951 | 574 |
| `word_sample.docx` | docx | MIT (docling test fixture) | 0.956 | 1.000 | 520 |
| `lorem_ipsum.docx` | docx | MIT (docling test fixture) | 1.000 | 1.000 | 1 |
| `unit_test_headers.docx` | docx | MIT (docling test fixture) | 1.000 | 1.000 | 3 |
| `word_tables.docx` | docx | MIT (docling test fixture) | 1.000 | 1.000 | 4 |
| `powerpoint_sample.pptx` | pptx | MIT (docling test fixture) | 0.848 | 0.989 | 134 |
| `xlsx_01.xlsx` | xlsx | MIT (docling test fixture) | 0.629 | 0.993 | 159 |
| `example_01.html` | html | MIT (docling test fixture) | 1.000 | 0.964 | 55 |
| `table_01.html` | html | MIT (docling test fixture) | 1.000 | 1.000 | 2 |

- The xlsx (0.629) and pptx (0.848) recall gaps come from the reference side: docling's Markdown repeats chart data series as extra tables, copies merged-cell values into every spanned cell and adds list markers. This parser emits each cell value once.
- Some PDF reference text carries LaTeX-style markup (for example `NH_3`) that the `\w+` tokeniser keeps as one token, which lowers PDF recall slightly.
- The PDF parser uses pdfplumber with `x_tolerance=1`, because the default merged words in tightly set papers.
- Licences: the arXiv papers are CC BY 4.0, except `arxiv-1602.06541v2` (CC BY-SA 4.0). The docling fixtures are MIT. The py-pdf/benchmarks ground truth is BSD-3-Clause.

### Chunking

Retrieval only, run on 2026-11-12. Each variant searches its own collection with dense retrieval only (`jinaai/jina-embeddings-v5-text-nano-retrieval`), so chunking is the only difference. A document counts once, at its best chunk, so recall@100 counts the distinct documents in the top 100 chunks and smaller chunks are penalised by construction. The `naive_rag` row is fixed 800-character chunks with no overlap. Full tables: [SciFact](../../../../results/scifact_chunking.md), [TVPL](../../../../results/tvpl_chunking.md). The other rows in the root README are untouched.

**SciFact** (300 queries, 5,183 documents)

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

**TVPL** (1,000 sampled queries, 10,576 documents)

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

- On both sets fixed 800 (`naive_rag`) is best or tied on nDCG@10: 0.751 against 0.750, 0.750 and 0.743 on SciFact, and 0.815 against 0.810, 0.802 and 0.794 on TVPL.
- A 200-character overlap does not help: `chunk_recursive_800_o200` against `chunk_recursive_800` is 0.750 against 0.750 on SciFact and 0.802 against 0.810 on TVPL.
- `chunk_recursive_400` has the lowest nDCG@10 on both sets.
- The `markdown` (heading-aware) strategy has no number here because neither set has Markdown headings. In the example, `index_files` uses the first heading as the title, so an indexed chunk carries it both as its title and in its heading path.
- Eval collections are reused when the stored point count matches, so delete the collection after changing chunking code.
- Chunking is compared on retrieval only. Answer metrics are not run, because the answer corpora are short passages; that is a roadmap item.

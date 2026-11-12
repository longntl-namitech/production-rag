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
- Both `recursive` and `markdown` accept an overlap: the next chunk starts with the tail of the previous one.
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
Pending: filled in from results/ after the eval runs.

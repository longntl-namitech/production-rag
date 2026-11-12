# Parsing quality

Run on 2026-11-12. 15 public documents: 7 arXiv PDFs scored against the py-pdf/benchmarks ground truth (commit `18ee8c4`), and 8 docling test fixtures (docx, pptx, xlsx, html; commit `a7760ab`).

Metric: token recall and token precision. Text is NFKC-normalised and casefolded, tokens are `\w+` matches, and both scores are multiset overlaps against the reference. Order and layout are not scored.

The docx, pptx, xlsx and html references are docling's own output, checked into its tests. For those files the score is agreement with another converter, not with a human transcription.

| Format | Files | Failed | Token recall | Token precision | p50 ms |
|---|---|---|---|---|---|
| pdf | 7 | 0 | 0.982 | 0.920 | 502 |
| docx | 4 | 0 | 0.989 | 1.000 | 4 |
| pptx | 1 | 0 | 0.848 | 0.989 | 134 |
| xlsx | 1 | 0 | 0.629 | 0.993 | 159 |
| html | 2 | 0 | 1.000 | 0.982 | 28 |
| all | 15 | 0 | 0.954 | 0.959 | 235 |

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

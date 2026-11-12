"""Parsing-quality eval on public PDF and Office files (zero LLM).

    uv run --extra parse python -m production_rag.methods.parse_and_chunk.parse_eval

Downloads 15 documents and their reference text (30 files, cached), parses each document and
scores it by token recall and precision against the reference. Writes `results/<name>.json`
(no document text) and `results/<name>.md`.
"""

import argparse
import json
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

from production_rag.core.eval.download import fetch
from production_rag.methods.parse_and_chunk.parsers import PARSERS

PYPDF_COMMIT = "18ee8c4e38a5627e9f2e56995e4bd57d0f2e4779"
DOCLING_COMMIT = "a7760abf0ca53321a3fb5ae9b1a65b584bb743b9"
LIBRARIES = [
    "pdfplumber",
    "python-docx",
    "python-pptx",
    "openpyxl",
    "markdownify",
    "beautifulsoup4",
]


@dataclass(frozen=True)
class ParseCase:
    name: str
    format: str
    source_url: str
    reference_url: str
    licence: str


def _pdf_case(arxiv_id: str, licence: str) -> ParseCase:
    base = "https://raw.githubusercontent.com/py-pdf/benchmarks"
    return ParseCase(
        name=f"arxiv-{arxiv_id}",
        format="pdf",
        source_url=f"https://arxiv.org/pdf/{arxiv_id}",
        reference_url=(
            f"{base}/{PYPDF_COMMIT}/read/extraction-ground-truth/{arxiv_id.split('v')[0]}.txt"
        ),
        licence=licence,
    )


def _docling_case(fmt: str, file: str) -> ParseCase:
    base = f"https://raw.githubusercontent.com/docling-project/docling/{DOCLING_COMMIT}/tests/data"
    return ParseCase(
        name=file,
        format=fmt,
        source_url=f"{base}/{fmt}/sources/{file}",
        reference_url=f"{base}/{fmt}/groundtruth/{file}.md",
        licence="MIT (docling test fixture)",
    )


CASES: list[ParseCase] = [
    *[
        _pdf_case(i, "CC BY 4.0")
        for i in [
            "1601.03642v1",
            "2201.00021v3",
            "2201.00022v2",
            "2201.00029v1",
            "2201.00069v1",
            "2201.00214v1",
        ]
    ],
    _pdf_case("1602.06541v2", "CC BY-SA 4.0"),
    *[
        _docling_case("docx", f)
        for f in [
            "word_sample.docx",
            "lorem_ipsum.docx",
            "unit_test_headers.docx",
            "word_tables.docx",
        ]
    ],
    _docling_case("pptx", "powerpoint_sample.pptx"),
    _docling_case("xlsx", "xlsx_01.xlsx"),
    _docling_case("html", "example_01.html"),
    _docling_case("html", "table_01.html"),
]


def tokens(text: str) -> Counter[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return Counter(re.findall(r"\w+", text))


def overlap_scores(reference: str, parsed: str) -> tuple[float, float]:
    """(recall, precision) as multiset overlaps; 0.0 when either side is empty."""
    ref, got = tokens(reference), tokens(parsed)
    n_ref, n_got = sum(ref.values()), sum(got.values())
    if not n_ref or not n_got:
        return 0.0, 0.0
    both = sum((ref & got).values())
    return both / n_ref, both / n_got


def score_case(case: ParseCase, source: Path, reference: Path) -> dict[str, Any]:
    """Parse by the case's known format, never by file suffix (cached files may have none)."""
    row: dict[str, Any] = {
        "name": case.name,
        "format": case.format,
        "licence": case.licence,
        "source_url": case.source_url,
        "reference_url": case.reference_url,
        "recall": None,
        "precision": None,
        "ms": None,
        "warnings": [],
        "error": None,
    }
    start = time.perf_counter()
    try:
        parsed = PARSERS[case.format](source)
    except Exception as e:
        row["ms"] = round((time.perf_counter() - start) * 1000)
        row["error"] = type(e).__name__
        return row
    row["ms"] = round((time.perf_counter() - start) * 1000)
    row["warnings"] = list(parsed.warnings)
    text = reference.read_text(encoding="utf-8", errors="replace")
    row["recall"], row["precision"] = overlap_scores(text, parsed.markdown)
    return row


def summarize(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["format"], []).append(row)
    groups["all"] = list(rows)
    out: dict[str, dict] = {}
    for name, group in groups.items():
        ok = [r for r in group if not r.get("error")]
        out[name] = {
            "files": len(group),
            "failed": len(group) - len(ok),
            "recall": statistics.mean(r["recall"] for r in ok) if ok else None,
            "precision": statistics.mean(r["precision"] for r in ok) if ok else None,
            "p50_ms": round(statistics.median(r["ms"] for r in group)) if group else None,
        }
    return out


def _frac(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def parse_table(rows: list[dict]) -> str:
    lines = [
        "| Document | Format | Licence | Token recall | Token precision | Parse ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            cells = ["failed: " + r["error"], "", str(r["ms"])]
        else:
            cells = [_frac(r["recall"]), _frac(r["precision"]), str(r["ms"])]
        lines.append(
            f"| `{r['name']}` | {r['format']} | {r['licence']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def summary_table(summary: dict[str, dict]) -> str:
    lines = [
        "| Format | Files | Failed | Token recall | Token precision | p50 ms |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['files']} | {s['failed']} | {_frac(s['recall'])} "
            f"| {_frac(s['precision'])} | {s['p50_ms']} |"
        )
    return "\n".join(lines)


def _header(meta: dict[str, Any]) -> str:
    return (
        "# Parsing quality\n\n"
        f"Run on {meta['date']}. 15 public documents: 7 arXiv PDFs scored against the "
        f"py-pdf/benchmarks ground truth (commit `{PYPDF_COMMIT[:7]}`), and 8 docling test "
        f"fixtures (docx, pptx, xlsx, html; commit `{DOCLING_COMMIT[:7]}`).\n\n"
        "Metric: token recall and token precision. Text is NFKC-normalised and casefolded, "
        "tokens are `\\w+` matches, and both scores are multiset overlaps against the reference. "
        "Order and layout are not scored.\n\n"
        "The docx, pptx, xlsx and html references are docling's own output, checked into its "
        "tests. For those files the score is agreement with another converter, not with a human "
        "transcription.\n\n"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="parse_quality")
    args = ap.parse_args(argv)
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        raise SystemExit("parse_eval needs the parse extra: uv sync --extra parse") from None
    rows = []
    for case in CASES:
        print(f"{case.name} ...", flush=True)
        rows.append(score_case(case, fetch(case.source_url), fetch(case.reference_url)))
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    meta = {
        "date": date.today().isoformat(),
        "pypdf_benchmarks_commit": PYPDF_COMMIT,
        "docling_commit": DOCLING_COMMIT,
        "libraries": versions,
    }
    summary = summarize(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2) + "\n"
    )
    (out / f"{args.name}.md").write_text(
        _header(meta) + summary_table(summary) + "\n\n" + parse_table(rows) + "\n"
    )
    print(summary_table(summary))


if __name__ == "__main__":
    main()

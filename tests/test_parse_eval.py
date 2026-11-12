import re
from collections import Counter

import pytest

from production_rag.methods.parse_and_chunk import parse_eval
from production_rag.methods.parse_and_chunk.parse_eval import (
    CASES,
    DOCLING_COMMIT,
    PYPDF_COMMIT,
    ParseCase,
    overlap_scores,
    parse_table,
    score_case,
    summarize,
    summary_table,
    tokens,
)
from production_rag.methods.parse_and_chunk.parsers import Parsed


def test_tokens_normalise():
    assert tokens("Ｆｕｌｌ <!-- image --> Café, café!") == Counter({"full": 1, "café": 2})


def test_overlap_scores():
    assert overlap_scores("a a b c", "a b b d") == (pytest.approx(2 / 4), pytest.approx(2 / 4))
    assert overlap_scores("a", "") == (0.0, 0.0)


def test_cases_are_pinned_and_licensed():
    assert len(CASES) == 15
    assert {c.format for c in CASES} == {"pdf", "docx", "pptx", "xlsx", "html"}
    assert all(PYPDF_COMMIT in c.reference_url or DOCLING_COMMIT in c.reference_url for c in CASES)
    pdfs = [c for c in CASES if c.format == "pdf"]
    assert len(pdfs) == 7
    assert all(re.search(r"/pdf/\d{4}\.\d{5}v\d$", c.source_url) for c in pdfs)
    assert all(c.licence for c in CASES)
    assert not any("wiki_duck" in c.source_url for c in CASES)


def test_summarize_excludes_failed():
    rows = [
        {"format": "pdf", "recall": 0.9, "precision": 0.8, "ms": 10, "error": None},
        {"format": "pdf", "recall": None, "precision": None, "ms": 5, "error": "PDFSyntaxError"},
    ]
    s = summarize(rows)["pdf"]
    assert (s["files"], s["failed"], s["recall"]) == (2, 1, 0.9)
    assert summarize(rows)["all"]["files"] == 2


def test_score_case_dispatches_by_format_not_suffix(tmp_path, monkeypatch):
    seen = []

    def fake_pdf(path):
        seen.append(path)
        return Parsed(markdown="alpha beta", format="pdf")

    monkeypatch.setitem(parse_eval.PARSERS, "pdf", fake_pdf)
    source = tmp_path / "0123456789abcdef-2201.00021v3"
    source.write_bytes(b"%PDF")
    ref = tmp_path / "ref.txt"
    ref.write_text("alpha beta gamma delta")
    case = ParseCase("p", "pdf", "https://arxiv.org/pdf/2201.00021v3", "u", "CC BY 4.0")
    row = score_case(case, source, ref)
    assert seen == [source]
    assert row["error"] is None
    assert (row["recall"], row["precision"]) == (0.5, 1.0)


def test_score_case_records_parser_error(tmp_path, monkeypatch):
    def boom(path):
        raise ValueError("bad")

    monkeypatch.setitem(parse_eval.PARSERS, "docx", boom)
    case = ParseCase("d", "docx", "u", "r", "MIT")
    row = score_case(case, tmp_path / "x", tmp_path / "y")
    assert row["error"] == "ValueError" and row["recall"] is None


def test_tables_render():
    rows = [
        {
            "name": "a",
            "format": "pdf",
            "licence": "L",
            "recall": 0.5,
            "precision": 1.0,
            "ms": 7,
            "error": None,
        },
        {
            "name": "b",
            "format": "pdf",
            "licence": "L",
            "recall": None,
            "precision": None,
            "ms": 3,
            "error": "ValueError",
        },
    ]
    assert parse_table(rows).splitlines()[0].startswith("| Document | Format | Licence |")
    assert "failed: ValueError" in parse_table(rows)
    assert "| pdf | 2 | 1 | 0.500 | 1.000 | 5 |" in summary_table(summarize(rows))

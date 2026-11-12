import pytest

from production_rag.core.chunking import (
    chunk_documents,
    fixed_size_chunks,
    markdown_section_chunks,
    recursive_chunks,
)


def test_chunks_respect_size_and_keep_words():
    text = " ".join(f"word{i}" for i in range(300))
    out = fixed_size_chunks(text, 50)
    assert all(len(c) <= 50 for c in out) and " ".join(out) == text


def test_long_word_is_cut():
    assert fixed_size_chunks("a" * 120, 50) == ["a" * 50, "a" * 50, "a" * 20]


def test_vietnamese_text_round_trips():
    text = "Người lao động được nghỉ phép năm hưởng nguyên lương theo hợp đồng"
    assert " ".join(fixed_size_chunks(text, 20)) == text


def test_blank_text_has_no_chunks():
    assert fixed_size_chunks("  \n ", 10) == []


def test_chunk_documents_indexes_per_doc():
    out = chunk_documents({"a": ("T", "x " * 40), "b": (None, "y"), "c": (None, " ")}, 20)
    a = [(c.doc_id, c.chunk_index) for c in out if c.doc_id == "a"]
    assert a == [("a", i) for i in range(4)]
    assert {c.title for c in out if c.doc_id == "a"} == {"T"}
    assert "c" not in {c.doc_id for c in out}
    assert [c.doc_id for c in out][-1] == "b"


LEGAL = (
    "Điều 1. Phạm vi\n1. Luật này quy định.\n2. Áp dụng cho mọi người.\n\n"
    "Điều 2. Giải thích\na) Người lao động là người làm việc."
)


def _norm(s):
    return " ".join(s.split())


def test_size_guards():
    for fn in (fixed_size_chunks, recursive_chunks, markdown_section_chunks):
        with pytest.raises(ValueError):
            fn("a b", 0)
    with pytest.raises(ValueError):
        recursive_chunks("a b", 10, overlap=10)


def test_recursive_respects_size_and_rebuilds_text():
    text = (LEGAL + "\n\n") * 20
    out = recursive_chunks(text, 120)
    assert all(len(c) <= 120 for c in out) and _norm(" ".join(out)) == _norm(text)


def test_recursive_prefers_line_boundaries():
    out = recursive_chunks(LEGAL, 60)
    assert out[0].startswith("Điều 1.") and all(not c.startswith("Phạm") for c in out)


def test_recursive_cuts_a_long_word():
    assert recursive_chunks("x" * 130, 50) == ["x" * 50, "x" * 50, "x" * 30]


def test_overlap_repeats_previous_tail_only():
    words = " ".join(f"w{i}" for i in range(200))
    out = recursive_chunks(words, 100, overlap=30)
    assert all(len(c) <= 100 for c in out)
    for prev, cur in zip(out, out[1:], strict=False):
        assert cur.split()[0] in prev.split()[-8:]
    assert _norm(" ".join(recursive_chunks(words, 100))) == _norm(words)


def test_markdown_sections_carry_heading_path():
    md = "intro text\n\n# A\n\nalpha\n\n## B\n\nbeta\n\n```\n# not a heading\n```\n\n# C\n"
    out = markdown_section_chunks(md, 200)
    assert out[0] == "intro text" and out[1] == "A\nalpha"
    assert out[2].startswith("A > B\nbeta") and "# not a heading" in out[2]
    assert not any(c.startswith("C") for c in out)


def test_chunk_documents_strategies():
    docs = {"d": ("T", LEGAL * 5)}
    assert chunk_documents(docs, 100) == chunk_documents(docs, 100, strategy="fixed")
    rec = chunk_documents(docs, 100, strategy="recursive", overlap=20)
    assert [c.chunk_index for c in rec] == list(range(len(rec)))
    assert {c.title for c in rec} == {"T"}
    with pytest.raises(ValueError):
        chunk_documents(docs, 100, overlap=20)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_documents({"d": (None, "a b")}, 100, strategy="nope")


def test_markdown_validates_arguments_even_without_body():
    with pytest.raises(ValueError):
        markdown_section_chunks("# only a heading", 0)
    with pytest.raises(ValueError):
        markdown_section_chunks("# only a heading", 100, overlap=100)


def test_markdown_heading_path_falls_back_to_last_heading():
    long_a = "A" * 60
    md = f"# {long_a}\n\n## B\n\nbody text\n"
    out = markdown_section_chunks(md, 150)
    assert out == ["B\nbody text"]
    with pytest.raises(ValueError):
        markdown_section_chunks(md, 100)

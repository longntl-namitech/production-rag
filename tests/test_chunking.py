from production_rag.core.chunking import chunk_documents, fixed_size_chunks


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

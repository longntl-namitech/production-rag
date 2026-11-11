import pytest

from production_rag.core.eval.datasets import (
    legal_corpus,
    load_answer_dataset,
    sample_ids,
    strip_context_prefix,
)


def test_sample_ids_prefix_stable():
    ids = [f"q{i}" for i in range(918)]
    assert sample_ids(ids, 20) == sample_ids(ids, 150)[:20]
    assert len(set(sample_ids(ids, 150))) == 150


def test_sample_ids_larger_than_pool():
    assert sorted(sample_ids(["b", "a"], 5)) == ["a", "b"]


def test_strip_context_prefix():
    assert strip_context_prefix("ngữ cảnh 3: 1. Điều khoản") == "1. Điều khoản"
    assert strip_context_prefix("không có tiền tố") == "không có tiền tố"


def test_legal_corpus_dedupes_across_positions():
    rows = [
        {"context": ["ngữ cảnh 1: A", "ngữ cảnh 2: B"]},
        {"context": ["ngữ cảnh 1: B", "ngữ cảnh 2: C"]},
    ]
    corpus = legal_corpus(rows)
    assert [t for _, t in corpus.values()] == ["A", "B", "C"]
    assert all(k.startswith("vl-") for k in corpus)


@pytest.mark.integration
@pytest.mark.parametrize(
    "name,questions,docs", [("mini_wiki", 918, 3190), ("vi_legal", 3176, 2125)]
)
def test_load_answer_dataset_sizes(name, questions, docs):
    ds = load_answer_dataset(name, sample=10_000)
    assert len(ds.questions) == questions
    assert len(ds.corpus) >= docs if name == "mini_wiki" else len(ds.corpus) == docs
    assert ds.gold.keys() == ds.questions.keys()

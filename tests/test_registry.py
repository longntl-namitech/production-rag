import pytest

from production_rag.core.registry import METHODS, get_method, legs_for


def test_registry_has_v1_methods():
    assert set(METHODS) == {
        "dense",
        "bm25_no_idf",
        "bm25",
        "bm25_vi_stopwords",
        "hybrid_rrf",
        "hybrid_dbsf",
        "hybrid_weighted_rrf",
        "hybrid_rrf_rerank",
    }


def test_weighted_rrf_has_two_weights():
    method = METHODS["hybrid_weighted_rrf"]
    assert method.fusion == "weighted_rrf"
    assert method.weights == (1.0, 0.5)
    assert len(method.weights) == len(method.legs)


def test_legs_for_search_type():
    assert legs_for(METHODS["hybrid_rrf"], "hybrid") == ("dense", "sparse")
    assert legs_for(METHODS["hybrid_rrf"], "sparse") == ("sparse",)
    assert legs_for(METHODS["bm25_vi_stopwords"], "sparse") == ("sparse_vi",)
    assert legs_for(METHODS["bm25"], "dense") == ("dense",)
    assert legs_for(METHODS["dense"], "sparse") == ("sparse",)


def test_unknown_method_lists_names():
    with pytest.raises(KeyError, match="hybrid_rrf"):
        get_method("nope")

import pytest
from qdrant_client import models

from production_rag.core.fusion import QueryVectors, build_query

SPARSE = models.SparseVector(indices=[1], values=[1.0])
SPARSE_VI = models.SparseVector(indices=[2], values=[1.0])
VECTORS = QueryVectors(dense=[0.1, 0.2], sparse=SPARSE, sparse_vi=SPARSE_VI)
FLT = models.Filter(must=[models.FieldCondition(key="tags", match=models.MatchValue(value="a"))])


def test_single_dense_leg():
    kwargs = build_query(("dense",), "none", None, VECTORS, 40, FLT)
    assert kwargs == {"query": [0.1, 0.2], "using": "dense", "query_filter": FLT}


def test_noidf_leg_uses_sparse_vector():
    kwargs = build_query(("sparse_noidf",), "none", None, VECTORS, 40, None)
    assert kwargs["query"] == SPARSE and kwargs["using"] == "sparse_noidf"


def test_vi_leg_uses_vi_vector():
    kwargs = build_query(("sparse_vi",), "none", None, VECTORS, 40, None)
    assert kwargs["query"] == SPARSE_VI


def test_rrf_has_two_prefetch_with_filter():
    kwargs = build_query(("dense", "sparse"), "rrf", None, VECTORS, 40, FLT)
    assert [p.using for p in kwargs["prefetch"]] == ["dense", "sparse"]
    assert all(p.limit == 40 and p.filter == FLT for p in kwargs["prefetch"])
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)


def test_dbsf_query():
    kwargs = build_query(("dense", "sparse"), "dbsf", None, VECTORS, 40, None)
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.DBSF)


def test_weighted_rrf_passes_weights():
    kwargs = build_query(("dense", "sparse"), "weighted_rrf", (1.0, 0.5), VECTORS, 40, None)
    assert kwargs["query"] == models.RrfQuery(rrf=models.Rrf(weights=[1.0, 0.5]))


def test_missing_vector_raises():
    with pytest.raises(ValueError):
        build_query(("dense",), "none", None, QueryVectors(None, SPARSE, None), 40, None)

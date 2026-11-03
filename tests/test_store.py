import pytest
from fakes import FakeEmbedder, FakeSparseEncoder
from qdrant_client import AsyncQdrantClient

from production_rag.core.fusion import QueryVectors, build_query
from production_rag.core.store import ChunkPoint, Store, build_filter, point_id

DOCS = [
    ("d1", 0, "solar panels on roofs", ["energy"]),
    ("d2", 0, "wind turbines offshore", ["energy"]),
    ("d3", 0, "bread baking at home", ["food"]),
]


async def make_points(docs=DOCS) -> list[ChunkPoint]:
    texts = [d[2] for d in docs]
    dense = await FakeEmbedder().embed_documents(texts)
    sparse = FakeSparseEncoder().encode_documents(texts)
    return [
        ChunkPoint(doc_id, idx, text, dense[i], sparse[i], sparse[i], tags=tags)
        for i, (doc_id, idx, text, tags) in enumerate(docs)
    ]


@pytest.fixture
async def store():
    s = Store(AsyncQdrantClient(":memory:"))
    await s.ensure_collection("c", 8)
    await s.upsert("c", await make_points())
    return s


async def _dense_query(text: str):
    [vec] = await FakeEmbedder().embed_queries([text])
    return QueryVectors(dense=vec, sparse=None, sparse_vi=None)


def test_point_id_deterministic():
    assert point_id("d1", 0) == point_id("d1", 0)
    assert point_id("d1", 0) != point_id("d1", 1)


async def test_upsert_twice_same_count(store):
    await store.upsert("c", await make_points())
    assert await store.count("c") == 3


async def test_ensure_collection_is_idempotent(store):
    await store.ensure_collection("c", 8)
    assert await store.count("c") == 3


async def test_search_dense_returns_rows_with_payload(store):
    kwargs = build_query(("dense",), "none", None, await _dense_query("solar panels"), 10, None)
    rows = await store.search("c", kwargs, limit=2, threshold=None)
    assert rows[0].doc_id == "d1"
    assert rows[0].text == "solar panels on roofs"
    assert rows[0].tags == ["energy"]
    assert len(rows) == 2


async def test_filter_scalar_and_any(store):
    vectors = await _dense_query("solar panels")
    food = build_query(("dense",), "none", None, vectors, 10, build_filter({"tags": "food"}))
    assert [r.doc_id for r in await store.search("c", food, limit=10, threshold=None)] == ["d3"]
    either = build_query(
        ("dense",), "none", None, vectors, 10, build_filter({"doc_id": ["d2", "d3"]})
    )
    found = {r.doc_id for r in await store.search("c", either, limit=10, threshold=None)}
    assert found == {"d2", "d3"}


def test_empty_filter_is_none():
    assert build_filter({}) is None


async def test_fetch_chunks_range_ordered():
    s = Store(AsyncQdrantClient(":memory:"))
    await s.ensure_collection("c", 8)
    await s.upsert("c", await make_points([("d", i, f"part {i}", []) for i in (2, 0, 1, 3)]))
    rows = await s.fetch_chunks("c", "d", 1, 2)
    assert [r.chunk_index for r in rows] == [1, 2]
    assert [r.chunk_index for r in await s.fetch_chunks("c", "d", None, None)] == [0, 1, 2, 3]


async def _hybrid_vectors(text: str) -> QueryVectors:
    [vec] = await FakeEmbedder().embed_queries([text])
    [sparse] = FakeSparseEncoder().encode_queries([text])
    return QueryVectors(dense=vec, sparse=sparse, sparse_vi=sparse)


@pytest.mark.parametrize(
    "fusion,weights", [("rrf", None), ("dbsf", None), ("weighted_rrf", (1.0, 0.5))]
)
async def test_hybrid_fusion_runs(store, fusion, weights):
    kwargs = build_query(
        ("dense", "sparse"), fusion, weights, await _hybrid_vectors("solar panels"), 10, None
    )
    rows = await store.search("c", kwargs, limit=3, threshold=None)
    assert rows[0].doc_id == "d1"

import pytest
from fakes import FakeEmbedder, FakeSparseEncoder
from qdrant_client import AsyncQdrantClient

from production_rag.core.config import Settings
from production_rag.core.deps import Deps
from production_rag.core.store import ChunkPoint, Store

CORPUS = [
    ("d1", 0, "solar panels on roofs", ["a"]),
    ("d2", 0, "wind turbines offshore", ["a"]),
    ("d3", 0, "bread baking at home", []),
    ("d4", 0, "river fishing guide", []),
    ("d5", 0, "mountain hiking boots", []),
    ("tea", 0, "history of tea", []),
    ("tea", 1, "tea ceremony in japan", []),
    ("tea", 2, "tea farming today", []),
]


@pytest.fixture
async def deps() -> Deps:
    embedder, sparse = FakeEmbedder(), FakeSparseEncoder()
    store = Store(AsyncQdrantClient(":memory:"))
    await store.ensure_collection("c", embedder.dim)
    texts = [d[2] for d in CORPUS]
    dense = await embedder.embed_documents(texts)
    vectors = sparse.encode_documents(texts)
    await store.upsert(
        "c",
        [
            ChunkPoint(doc_id, idx, text, dense[i], vectors[i], vectors[i], tags=tags)
            for i, (doc_id, idx, text, tags) in enumerate(CORPUS)
        ],
    )
    embedder.calls = sparse.calls = 0
    return Deps(
        settings=Settings(),
        store=store,
        embedder=embedder,
        bm25=sparse,
        bm25_vi=FakeSparseEncoder(),
    )

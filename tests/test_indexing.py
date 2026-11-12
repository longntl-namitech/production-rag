from fakes import FakeEmbedder, FakeSparseEncoder
from qdrant_client import AsyncQdrantClient

from production_rag.core.config import Settings
from production_rag.core.deps import Deps
from production_rag.core.eval.datasets import subsample
from production_rag.core.indexing import Chunk, index_chunks
from production_rag.core.store import Store

CHUNKS = [Chunk(f"d{i}", 0, f"text number {i}", f"title {i}" if i % 2 else None) for i in range(5)]


def make_deps() -> Deps:
    return Deps(
        settings=Settings(),
        store=Store(AsyncQdrantClient(":memory:")),
        embedder=FakeEmbedder(),
        bm25=FakeSparseEncoder(),
        bm25_vi=FakeSparseEncoder(),
    )


async def test_index_chunks_counts():
    deps = make_deps()
    assert await index_chunks(deps, "toy", CHUNKS, batch_size=2) == 5
    assert deps.embedder.calls == 3


async def test_reindex_skips_embedding():
    deps = make_deps()
    await index_chunks(deps, "toy", CHUNKS, batch_size=2)
    calls = deps.embedder.calls
    assert await index_chunks(deps, "toy", CHUNKS, batch_size=2) == 5
    assert deps.embedder.calls == calls


def test_subsample_is_deterministic():
    ids = [f"q{i}" for i in range(50)]
    assert subsample(ids, 10, 13) == subsample(list(reversed(ids)), 10, 13)
    assert len(subsample(ids, 10, 13)) == 10
    assert subsample(ids, 100, 13) == sorted(ids)

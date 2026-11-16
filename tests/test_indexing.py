from fakes import FakeEmbedder, FakeSparseEncoder
from qdrant_client import AsyncQdrantClient

from production_rag.core.config import Settings
from production_rag.core.deps import Deps
from production_rag.core.eval.datasets import subsample
from production_rag.core.indexing import Chunk, _document, index_chunks
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


def test_document_without_context_is_unchanged():
    assert _document("T", "x") == "T\nx"
    assert _document(None, "x") == "x"


def test_document_with_context():
    assert _document("T", "x", "C") == "T\nC\nx"
    assert _document(None, "x", "C") == "C\nx"


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    async def embed_documents(self, texts):
        self.texts += texts
        return await super().embed_documents(texts)


async def test_index_chunks_embeds_context_but_stores_raw_text():
    deps = make_deps()
    deps.embedder = RecordingEmbedder()
    chunk = Chunk("d1", 0, "annual leave is twelve days", "Leave", "From the HR leave policy.")
    await index_chunks(deps, "ctx", [chunk])
    assert deps.embedder.texts == ["Leave\nFrom the HR leave policy.\nannual leave is twelve days"]
    [row] = await deps.store.fetch_chunks("ctx", "d1", None, None)
    assert row.text == "annual leave is twelve days"


def test_subsample_is_deterministic():
    ids = [f"q{i}" for i in range(50)]
    assert subsample(ids, 10, 13) == subsample(list(reversed(ids)), 10, 13)
    assert len(subsample(ids, 10, 13)) == 10
    assert subsample(ids, 100, 13) == sorted(ids)

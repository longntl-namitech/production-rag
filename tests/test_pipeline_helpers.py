from qdrant_client import AsyncQdrantClient
from test_store import make_points

from production_rag.core.models import ContextOption, Row
from production_rag.core.pipeline.context import expand_context
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.merge import merge_max
from production_rag.core.store import Store


def row(id: str, score: float, doc_id: str = "d", chunk_index: int = 0, text: str = "") -> Row:
    return Row(id=id, doc_id=doc_id, chunk_index=chunk_index, text=text, score=score)


def test_deadline_with_fake_clock():
    now = [100.0]
    deadline = Deadline(500, clock=lambda: now[0])
    assert deadline.remaining_ms() == 500
    now[0] += 0.2
    assert deadline.remaining_ms() == 300
    assert abs(deadline.remaining_s() - 0.3) < 1e-9
    assert not deadline.expired()
    now[0] += 0.4
    assert deadline.remaining_ms() == 0
    assert deadline.expired()


def test_merge_max_keeps_best_score_per_id():
    merged = merge_max([[row("a", 0.2), row("b", 0.5)], [row("a", 0.9)]], k=10)
    assert [(r.id, r.score) for r in merged] == [("a", 0.9), ("b", 0.5)]


def test_merge_max_truncates_to_k():
    assert len(merge_max([[row(str(i), i) for i in range(5)]], k=3)) == 3


async def _three_chunk_store() -> Store:
    store = Store(AsyncQdrantClient(":memory:"))
    await store.ensure_collection("c", 8)
    await store.upsert("c", await make_points([("d", i, f"part {i}", []) for i in range(3)]))
    return store


async def test_expand_prev_next():
    store = await _three_chunk_store()
    hit = row("x", 0.7, chunk_index=1, text="part 1")
    [expanded] = await expand_context(store, "c", [hit], ContextOption(prev=1, next=1))
    assert expanded.text == "part 0\npart 1\npart 2"
    assert (expanded.id, expanded.score) == ("x", 0.7)


async def test_expand_full_document():
    store = await _three_chunk_store()
    hit = row("x", 0.7, chunk_index=2, text="part 2")
    [expanded] = await expand_context(store, "c", [hit], ContextOption(full=True))
    assert expanded.text == "part 0\npart 1\npart 2"

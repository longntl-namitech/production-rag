import asyncio

from production_rag.core.models import SearchRequest
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search


def request(**kwargs) -> SearchRequest:
    return SearchRequest(collection="c", **{"query": "solar panels", **kwargs})


async def test_embed_once_for_two_specs(deps):
    req = request(specs=[{"name": "a", "filter": {"tags": "a"}}, {"name": "b"}])
    response = await run_search(req, deps, Deadline(4000))
    assert deps.embedder.calls == 1
    assert deps.bm25.calls == 1
    assert {name: r.status for name, r in response.results.items()} == {"a": "ok", "b": "ok"}
    assert {r.doc_id for r in response.results["a"].rows} <= {"d1", "d2"}
    assert response.method == "hybrid_rrf"


async def test_dense_only_spec_skips_sparse_encoding(deps):
    response = await run_search(request(method="dense"), deps, Deadline(4000))
    assert deps.bm25.calls == 0
    assert response.results["default"].rows[0].doc_id == "d1"


async def test_vi_method_uses_vi_encoder(deps):
    await run_search(request(method="bm25_vi_stopwords"), deps, Deadline(4000))
    assert deps.bm25_vi.calls == 1
    assert deps.bm25.calls == 0


async def test_failed_spec_isolated(deps, monkeypatch):
    original = deps.store.search

    async def search(name, kwargs, limit, threshold):
        if "boom" in repr(kwargs):
            raise RuntimeError("secret detail")
        return await original(name, kwargs, limit, threshold)

    monkeypatch.setattr(deps.store, "search", search)
    req = request(specs=[{"name": "bad", "filter": {"tags": "boom"}}, {"name": "good"}])
    response = await run_search(req, deps, Deadline(4000))
    bad, good = response.results["bad"], response.results["good"]
    assert (bad.status, bad.error, bad.rows) == ("failed", "RuntimeError", [])
    assert good.status == "ok" and good.rows


async def test_slow_spec_times_out_as_failed(deps, monkeypatch):
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(deps.store, "search", slow)
    response = await run_search(request(), deps, Deadline(50))
    assert response.results["default"].status == "failed"
    assert response.results["default"].error == "TimeoutError"


async def test_multi_query_merged(deps):
    req = request(query=["solar panels", "bread baking"], specs=[{"name": "s", "k": 2}])
    response = await run_search(req, deps, Deadline(4000))
    assert {r.doc_id for r in response.results["s"].rows} == {"d1", "d3"}


async def test_context_expansion_applied(deps):
    req = request(
        query="tea ceremony japan",
        method="dense",
        specs=[{"name": "s", "k": 1, "context": {"prev": 1, "next": 1}}],
    )
    [row] = (await run_search(req, deps, Deadline(4000))).results["s"].rows
    assert row.text == "history of tea\ntea ceremony in japan\ntea farming today"


async def test_slow_embedding_bounded_by_deadline(deps, monkeypatch):
    import pytest

    from production_rag.core.pipeline.search import DeadlineExpired

    async def slow(texts):
        await asyncio.sleep(1)

    monkeypatch.setattr(deps.embedder, "embed_queries", slow)
    with pytest.raises(DeadlineExpired):
        await asyncio.wait_for(run_search(request(), deps, Deadline(50)), 0.5)

import asyncio

import httpx
import pytest

from production_rag.core.models import Row, SearchRequest
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.rerank_stage import apply_rerank
from production_rag.core.pipeline.search import run_search
from production_rag.core.rerankers.cohere_http import CohereHttpReranker
from production_rag.methods import get_variant


def rows(*ids: str) -> list[Row]:
    return [Row(id=i, doc_id=i, text=f"text {i}", score=1.0 - n / 10) for n, i in enumerate(ids)]


class ScoreByText:
    """Scores 'text c' highest, then b, then a."""

    def __init__(self):
        self.calls = 0

    async def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return [{"text a": 0.1, "text b": 0.5, "text c": 0.9}.get(t, 0.0) for t in texts]


class Raising:
    async def score(self, query, texts):
        raise RuntimeError("down")


class Slow:
    async def score(self, query, texts):
        await asyncio.sleep(1)
        return [0.0] * len(texts)


async def test_rerank_reorders():
    out, status, ms = await apply_rerank(
        ScoreByText(), "q", rows("a", "b", "c"), Deadline(4000), 300
    )
    assert [r.id for r in out] == ["c", "b", "a"]
    assert out[0].score == 0.9
    assert status == "ok" and ms is not None


async def test_rerank_off_without_reranker():
    out, status, ms = await apply_rerank(None, "q", rows("a", "b"), Deadline(4000), 300)
    assert ([r.id for r in out], status, ms) == (["a", "b"], "off", None)


async def test_rerank_skipped_when_budget_low():
    reranker = ScoreByText()
    out, status, _ = await apply_rerank(reranker, "q", rows("a", "b", "c"), Deadline(100), 300)
    assert status == "skipped_budget"
    assert [r.id for r in out] == ["a", "b", "c"]
    assert reranker.calls == 0


async def test_rerank_failure_keeps_order():
    out, status, _ = await apply_rerank(Raising(), "q", rows("a", "b", "c"), Deadline(4000), 300)
    assert (status, [r.id for r in out]) == ("failed", ["a", "b", "c"])


async def test_rerank_timeout_keeps_order():
    out, status, _ = await apply_rerank(Slow(), "q", rows("a", "b"), Deadline(400), 300)
    assert (status, [r.id for r in out]) == ("failed", ["a", "b"])


async def test_trivial_pool_skipped():
    _, status, _ = await apply_rerank(ScoreByText(), "q", rows("a"), Deadline(4000), 300)
    assert status == "skipped_trivial"


async def test_cohere_http_maps_indexes():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = CohereHttpReranker("http://x/v1", "k", "m", client=client)
    assert await reranker.score("q", ["a", "b"]) == [0.2, 0.9]


async def test_pool_widened_for_rerank(deps, monkeypatch):
    limits = []
    original = deps.store.search

    async def search(name, kwargs, limit, threshold):
        limits.append(limit)
        return await original(name, kwargs, limit, threshold)

    monkeypatch.setattr(deps.store, "search", search)
    deps.reranker = ScoreByText()
    req = SearchRequest(
        query="solar", collection="c", specs=[{"name": "s", "k": 10, "rerank": True}]
    )
    await run_search(req, get_variant("hybrid_rrf"), deps, Deadline(4000))
    assert limits == [40]


async def test_method_default_rerank_and_spec_override(deps):
    deps.reranker = ScoreByText()
    req = SearchRequest(
        query="solar panels",
        collection="c",
        specs=[{"name": "default_on", "k": 3}, {"name": "off", "k": 3, "rerank": False}],
    )
    response = await run_search(req, get_variant("hybrid_rrf_rerank"), deps, Deadline(4000))
    assert response.results["default_on"].rerank == "ok"
    assert response.results["off"].rerank == "off"
    assert len(response.results["default_on"].rows) == 3


@pytest.mark.integration
async def test_cross_encoder_scores_relevant_higher():
    from production_rag.core.config import Settings
    from production_rag.core.rerankers.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(Settings().cross_encoder_model)
    scores = await reranker.score(
        "mức phạt vượt đèn đỏ", ["Mức phạt khi vượt đèn đỏ với xe máy", "Thủ tục đăng ký kết hôn"]
    )
    assert scores[0] > scores[1]


async def test_slow_rerank_with_context_still_returns_rows(deps, monkeypatch):
    original = deps.store.fetch_chunks

    async def slow_fetch(*args):
        await asyncio.sleep(0.1)
        return await original(*args)

    monkeypatch.setattr(deps.store, "fetch_chunks", slow_fetch)
    deps.reranker = Slow()
    req = SearchRequest(
        query="tea ceremony japan",
        collection="c",
        specs=[{"name": "s", "k": 2, "rerank": True, "context": {"prev": 1, "next": 1}}],
    )
    result = (await run_search(req, get_variant("hybrid_rrf"), deps, Deadline(500))).results["s"]
    assert result.status == "ok"
    assert result.rerank == "failed"
    assert result.rows


async def test_rerank_sees_title_and_text():
    seen = []

    class Recording:
        async def score(self, query, texts):
            seen.extend(texts)
            return [0.0] * len(texts)

    rows_ = [Row(id="a", doc_id="a", text="body", title="Title", score=1.0), *rows("b")]
    await apply_rerank(Recording(), "q", rows_, Deadline(4000), 300)
    assert seen == ["Title\nbody", "text b"]


async def test_rerank_tiny_min_budget_skips():
    r = ScoreByText()
    _, status, _ = await apply_rerank(r, "q", rows("a", "b"), Deadline(30), 0)
    assert status == "skipped_budget" and r.calls == 0


async def test_rerank_wrong_score_count_keeps_order():
    class Short:
        async def score(self, query, texts):
            return [0.9]

    out, status, _ = await apply_rerank(Short(), "q", rows("a", "b"), Deadline(4000), 300)
    assert (status, [r.id for r in out]) == ("failed", ["a", "b"])


async def test_rerank_requested_without_reranker_keeps_k(deps, monkeypatch):
    limits = []
    original = deps.store.search

    async def search(name, kwargs, limit, threshold):
        limits.append(limit)
        return await original(name, kwargs, limit, threshold)

    monkeypatch.setattr(deps.store, "search", search)
    assert deps.reranker is None
    req = SearchRequest(
        query="solar", collection="c", specs=[{"name": "s", "k": 10, "rerank": True}]
    )
    res = (await run_search(req, get_variant("hybrid_rrf"), deps, Deadline(4000))).results["s"]
    assert limits == [10] and res.rerank == "off"


async def test_rerank_import_error_propagates():
    class Missing:
        async def score(self, query, texts):
            raise ImportError("needs the local-models extra")

    with pytest.raises(ImportError):
        await apply_rerank(Missing(), "q", rows("a", "b"), Deadline(4000), 300)

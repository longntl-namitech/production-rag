import json

import pytest
from fakes import FakeLLM, text_row
from test_contextual_headers import Boom, BudgetAfter

from production_rag.core.llm import BudgetExceeded, Usage
from production_rag.core.models import SpecResult
from production_rag.methods import METHOD_NAMES
from production_rag.methods.llm_rerank import pipeline as llm_rerank
from production_rag.methods.llm_rerank.rerank import rerank, rerank_messages

QUESTION = "how do solar panels make power"


def rows():
    return [
        text_row("Solar panels turn light into power.", title="Solar", doc_id="d1", score=0.9),
        text_row("Bread rises slowly.", doc_id="d2", score=0.5),
        text_row("Panels sit on roofs.", doc_id="d3", score=0.4),
    ]


def scores(*pairs: tuple[int, object]) -> str:
    return json.dumps({"scores": [{"id": i, "score": s} for i, s in pairs]})


def reply(text: str, **kwargs) -> FakeLLM:
    return FakeLLM(lambda m: text, **kwargs)


async def test_rerank_sorts_by_score():
    out, usage = await rerank(reply(scores((1, 2), (2, 9), (3, 5))), QUESTION, rows())
    assert [(r.doc_id, r.score, r.title) for r in out] == [
        ("d2", 0.5, None),
        ("d3", 0.4, None),
        ("d1", 0.9, "Solar"),
    ]
    assert usage == Usage(10, 5)


async def test_rerank_ties_keep_base_order():
    out, _ = await rerank(reply(scores((1, 4), (2, 4), (3, 4))), QUESTION, rows())
    assert [r.doc_id for r in out] == ["d1", "d2", "d3"]


UNUSABLE = [
    "nope",
    scores((1, 5), (2, 5)),
    scores((1, 5), (2, 5), (2, 5)),
    scores((0, 5), (2, 5), (3, 5)),
    scores((1, 5), (2, 5), (99, 5)),
    scores((1, 11), (2, 5), (3, 5)),
    scores((1, -1), (2, 5), (3, 5)),
    scores((1, True), (2, 5), (3, 5)),
    scores((1, "7"), (2, 5), (3, 5)),
    json.dumps({"scores": "none"}),
    json.dumps([1, 2, 3]),
]


@pytest.mark.parametrize("text", UNUSABLE)
async def test_rerank_unusable_reply_returns_base_order(text):
    original = rows()
    out, _ = await rerank(reply(text), QUESTION, original)
    assert out == original


async def test_rerank_cut_reply_or_failed_call_returns_base_order():
    original = rows()
    for llm in (reply(scores((1, 1), (2, 9), (3, 5)), finish_reason="length"), Boom()):
        out, _ = await rerank(llm, QUESTION, original)
        assert out == original


async def test_rerank_empty_rows_makes_no_call():
    llm = reply(scores())
    out, usage = await rerank(llm, QUESTION, [])
    assert out == [] and usage == Usage() and llm.calls == []


async def test_rerank_propagates_budget_exceeded():
    with pytest.raises(BudgetExceeded):
        await rerank(BudgetAfter(0), QUESTION, rows())


def test_rerank_messages_number_passages():
    text = rerank_messages(QUESTION, rows())[-1]["content"]
    assert QUESTION in text
    for marker in ("Passage 1:", "Passage 3:", "Bread rises slowly."):
        assert marker in text
    assert "Passage 4:" not in text


async def test_search_reranks_candidates_and_keeps_k(monkeypatch):
    asked = {}

    async def fake_search(question, collection, deps, *, k, deadline_ms):
        asked.update(question=question, k=k)
        candidates = [text_row(f"t{i}", doc_id=f"d{i}") for i in range(1, 21)]
        return SpecResult(status="ok", rows=candidates, elapsed_ms=1)

    monkeypatch.setattr(llm_rerank.naive_rag, "search", fake_search)
    llm = reply(scores(*[(i, 10 if i == 20 else 0) for i in range(1, 21)]))
    result, usage = await llm_rerank.search("q" * 2500, "c", None, llm, k=5)
    assert asked == {"question": "q" * 2000, "k": llm_rerank.CANDIDATES}
    assert [r.doc_id for r in result.rows] == ["d20", "d1", "d2", "d3", "d4"]
    assert usage == Usage(10, 5)


async def test_search_without_rows_makes_no_call(monkeypatch):
    async def fake_search(question, collection, deps, *, k, deadline_ms):
        return SpecResult(status="ok", rows=[], elapsed_ms=1)

    monkeypatch.setattr(llm_rerank.naive_rag, "search", fake_search)
    llm = reply(scores())
    result, usage = await llm_rerank.search("q", "c", None, llm)
    assert result.rows == [] and usage == Usage() and llm.calls == []


def test_llm_rerank_registered():
    assert "llm_rerank" in METHOD_NAMES
    assert llm_rerank.VARIANTS == {} and llm_rerank.DEFAULT is None

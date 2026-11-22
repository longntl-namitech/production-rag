import json

import pytest
from fakes import FakeLLM, text_row
from test_contextual_headers import Boom, BudgetAfter

from production_rag.core.llm import BudgetExceeded, Usage
from production_rag.core.models import SpecResult
from production_rag.methods import METHOD_NAMES
from production_rag.methods.retrieval_sufficiency import pipeline as retrieval_sufficiency
from production_rag.methods.retrieval_sufficiency.judge import (
    judge_sufficiency,
    sufficiency_messages,
)

QUESTION = "how do solar panels make power"


def rows():
    return [
        text_row("Solar panels turn light into power.", doc_id="d1"),
        text_row("Bread rises slowly.", doc_id="d2"),
    ]


def reply(text: str, **kwargs) -> FakeLLM:
    return FakeLLM(lambda m: text, **kwargs)


def label(value: str) -> FakeLLM:
    return reply(json.dumps({"label": value}))


@pytest.mark.parametrize("value", ["sufficient", "insufficient"])
async def test_judge_returns_the_label(value):
    got, usage = await judge_sufficiency(label(value), QUESTION, rows())
    assert got == value and usage == Usage(10, 5)


@pytest.mark.parametrize(
    "text",
    ["nope", json.dumps({"label": "maybe"}), json.dumps({"label": 1}), json.dumps(["sufficient"])],
)
async def test_judge_unusable_reply_has_no_label(text):
    got, _ = await judge_sufficiency(reply(text), QUESTION, rows())
    assert got is None


async def test_judge_cut_reply_or_failed_call_has_no_label():
    for llm in (reply(json.dumps({"label": "sufficient"}), finish_reason="length"), Boom()):
        got, _ = await judge_sufficiency(llm, QUESTION, rows())
        assert got is None


async def test_judge_empty_rows_is_insufficient_without_a_call():
    llm = label("sufficient")
    got, usage = await judge_sufficiency(llm, QUESTION, [])
    assert got == "insufficient" and usage == Usage() and llm.calls == []


async def test_judge_propagates_budget_exceeded():
    with pytest.raises(BudgetExceeded):
        await judge_sufficiency(BudgetAfter(0), QUESTION, rows())


def test_messages_hold_question_and_passages():
    text = sufficiency_messages(QUESTION, rows())[-1]["content"]
    assert QUESTION in text and "Passage 2:" in text and "Bread rises slowly." in text


async def test_search_returns_rows_and_label(deps):
    got = await retrieval_sufficiency.search(
        "solar panels", "c", deps, label("sufficient"), base="dense", k=2
    )
    assert got.label == "sufficient" and got.error is None
    assert len(got.rows) == 2 and got.usage == Usage(10, 5)


async def test_search_unusable_reply_keeps_rows_without_label(deps):
    got = await retrieval_sufficiency.search(
        "solar panels", "c", deps, reply("nope"), base="dense", k=2
    )
    assert got.label is None and got.error == "unusable reply" and len(got.rows) == 2


async def test_search_cuts_long_question_to_2000(monkeypatch):
    asked = {}

    async def fake_run_variant(variants, variant, query, collection, deps, k, deadline_ms):
        asked["query"] = query
        return SpecResult(status="ok", rows=[], elapsed_ms=1)

    monkeypatch.setattr(retrieval_sufficiency, "run_variant", fake_run_variant)
    got = await retrieval_sufficiency.search("q" * 2500, "c", None, label("sufficient"))
    assert asked["query"] == "q" * 2000
    assert got.label == "insufficient" and got.rows == []


async def test_search_retrieval_failure_has_no_label(monkeypatch):
    async def fake_run_variant(variants, variant, query, collection, deps, k, deadline_ms):
        return SpecResult(status="failed", elapsed_ms=1, error="boom")

    monkeypatch.setattr(retrieval_sufficiency, "run_variant", fake_run_variant)
    llm = label("sufficient")
    got = await retrieval_sufficiency.search("q", "c", None, llm)
    assert got.label is None and got.rows == [] and got.error == "boom" and llm.calls == []


def test_retrieval_sufficiency_registered():
    assert "retrieval_sufficiency" in METHOD_NAMES
    assert retrieval_sufficiency.VARIANTS == {} and retrieval_sufficiency.DEFAULT is None
    assert retrieval_sufficiency.BASE == "hybrid_weighted_rrf"

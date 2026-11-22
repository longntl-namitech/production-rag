import json

import pytest
from fakes import FakeLLM, text_row
from test_contextual_headers import BudgetAfter

from production_rag.methods.retrieval_sufficiency.eval import (
    estimate_query,
    parse_args,
    report_md,
    run_queries,
    summarize,
)
from production_rag.methods.retrieval_sufficiency.judge import MAX_TOKENS

ITEM_KEYS = {"query_id", "predicted", "gold_hit", "status", "error", "tokens"}
QUERIES = {"q1": "solar panels", "q2": "bread baking"}
GOLD = {"q1": ["d1"], "q2": ["d3"]}


def label(value: str) -> FakeLLM:
    return FakeLLM(lambda m: json.dumps({"label": value}))


async def run(llm, **kwargs):
    return await run_queries(
        deps=kwargs.pop("deps"),
        collection="c",
        queries=QUERIES,
        gold_docs=GOLD,
        llm=llm,
        base="dense",
        k=2,
        concurrency=1,
        **kwargs,
    )


async def test_run_queries_items_have_label_and_gold_hit(deps):
    items, complete = await run(label("sufficient"), deps=deps)
    assert complete and [i["query_id"] for i in items] == ["q1", "q2"]
    assert all(set(i) == ITEM_KEYS for i in items)
    assert [(i["predicted"], i["status"], i["gold_hit"]) for i in items] == [
        ("sufficient", "ok", True),
        ("sufficient", "ok", True),
    ]
    assert "solar panels" not in json.dumps(items)


async def test_run_queries_unusable_label_is_failed_not_insufficient(deps):
    items, _ = await run(FakeLLM(lambda m: "nope"), deps=deps)
    assert [(i["predicted"], i["status"], i["error"]) for i in items] == [
        ("failed", "failed", "unusable reply")
    ] * 2


async def test_run_queries_budget_stops_the_run(deps):
    items, complete = await run(BudgetAfter(1), deps=deps)
    assert not complete and len(items) == 1


def _item(qid, predicted, hit, status="ok"):
    return {
        "query_id": qid,
        "predicted": predicted,
        "gold_hit": hit,
        "status": status,
        "error": None,
        "tokens": 0,
    }


def test_summarize_agreement_excludes_failed():
    items = [
        _item("1", "sufficient", True),
        _item("2", "sufficient", False),
        _item("3", "insufficient", False),
        _item("4", "insufficient", True),
        _item("5", "failed", True, status="failed"),
    ]
    s = summarize(items)
    assert s["queries"] == 5 and s["failed"] == 1
    assert s["agreement"] == pytest.approx(0.5)
    assert s["sufficient_rate"] == pytest.approx(0.5)
    assert s["confusion"] == {
        "sufficient": {"hit": 1, "miss": 1},
        "insufficient": {"hit": 1, "miss": 1},
    }


def test_summarize_without_answers_has_no_rates():
    s = summarize([_item("1", "failed", False, status="failed")])
    assert s["agreement"] is None and s["sufficient_rate"] is None


def test_report_md_names_the_proxy_and_has_no_queries():
    items = [_item("1", "sufficient", True)]
    meta = {
        "dataset": "scifact",
        "date": "2026-01-01",
        "requested": 1,
        "queries": 1,
        "documents": 5,
        "base": "dense",
        "k": 5,
        "model": "m",
        "complete": True,
    }
    md = report_md({"meta": meta, "summary": summarize(items), "per_item": items})
    assert "proxy" in md and "Agreement" in md


def test_estimate_query_includes_output_cap():
    assert estimate_query("why", [text_row("a passage")]) > MAX_TOKENS


def test_parse_args_needs_a_cap_unless_estimating():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "scifact"])
    assert parse_args(["--dataset", "scifact", "--estimate-only"]).estimate_only
    assert parse_args(["--dataset", "tvpl", "--max-tokens", "5"]).max_tokens == 5

import asyncio
import json

import httpx
import pytest
from fakes import FakeLLM, text_row

from production_rag.core.llm import AuthFailed, OpenAIChat, TokenBudget
from production_rag.methods.fact_check import eval as fc_eval
from production_rag.methods.fact_check.eval import (
    GOLD_LABELS,
    PREDICTED,
    agreement,
    compare_md,
    estimate_claim,
    main,
    parse_args,
    preflight,
    report_md,
    run_claims,
    summarize_verdicts,
)

ITEM_KEYS = {
    "claim_id",
    "gold",
    "predicted",
    "status",
    "error",
    "doc_ids",
    "evidence_hit",
    "ms",
    "tokens",
    "reasoning",
}


def _item(claim_id, gold, predicted, hit, status="ok"):
    return {
        "claim_id": claim_id,
        "gold": gold,
        "predicted": predicted,
        "status": status,
        "error": None if status == "ok" else "bad_output",
        "doc_ids": [],
        "evidence_hit": hit,
        "ms": 1,
        "tokens": 0,
        "reasoning": 0,
    }


ITEMS = [
    _item("1", "supported", "supported", True),
    _item("2", "supported", "unverified", False),
    _item("3", "refuted", "refuted", True),
    _item("4", "refuted", "supported", True),
    _item("5", "unverified", "unverified", False),
    _item("6", "unverified", "failed", False, status="failed"),
]
META = {
    "kind": "fact_check",
    "date": "2026-01-01",
    "claims": 6,
    "documents": 5183,
    "base": "dense",
    "k": 5,
    "model": "m",
    "complete": True,
}


def mock_chat(budget=None, status=200) -> OpenAIChat:
    """Every call costs 100 tokens and says supported; a non-200 status refuses every call."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            error = {"code": "invalid_api_key", "message": "bad key sk-test"}
            return httpx.Response(status, json={"error": error})
        content = json.dumps({"label": "supported", "reason": "r"})
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 90, "completion_tokens": 10},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChat("sk-test", client=client, budget=budget, backoff_s=0)


def test_summarize_verdicts():
    s = summarize_verdicts(ITEMS)
    assert (s["claims"], s["failed"]) == (6, 1) and s["accuracy"] == pytest.approx(3 / 5)
    assert s["per_label"]["supported"] == {
        "gold": 2,
        "predicted": 2,
        "correct": 1,
        "precision": 0.5,
        "recall": 0.5,
    }
    # gold counts every claim, precision and recall only the answered ones
    assert s["per_label"]["unverified"] == {
        "gold": 2,
        "predicted": 2,
        "correct": 1,
        "precision": 0.5,
        "recall": 1.0,
    }
    assert s["confusion"]["unverified"] == {
        "supported": 0,
        "refuted": 0,
        "unverified": 1,
        "failed": 1,
    }
    assert sum(sum(r.values()) for r in s["confusion"].values()) == 6
    assert s["evidence_hit_rate"] == pytest.approx(3 / 4)
    assert (s["accuracy_hit"], s["accuracy_miss"]) == (pytest.approx(2 / 3), 0.0)
    assert s["unverified_share"] == pytest.approx(2 / 5)


def test_labels_are_three_gold_and_four_predicted():
    assert GOLD_LABELS == ("supported", "refuted", "unverified")
    assert PREDICTED == (*GOLD_LABELS, "failed")


def test_summary_of_nothing_has_no_ratios():
    s = summarize_verdicts([])
    assert s["claims"] == 0 and s["accuracy"] is None and s["evidence_hit_rate"] is None


def test_agreement_uses_claims_ok_in_both():
    b = [{**i, "predicted": "supported"} if i["claim_id"] == "2" else i for i in ITEMS]
    g = agreement(ITEMS, b)
    assert g["claims"] == 5 and g["same_label"] == pytest.approx(4 / 5)
    assert g["accuracy_a"] == pytest.approx(3 / 5) and g["accuracy_b"] == pytest.approx(4 / 5)


async def test_run_claims_records_hit_and_failures(deps):
    llm = FakeLLM(
        lambda m: '{"label": "supported", "reason": "r"}' if "solar" in m[-1]["content"] else "bad"
    )
    items, complete = await run_claims(
        deps=deps,
        collection="c",
        claims={"a": "solar panels", "b": "bread baking"},
        labels={"a": "supported", "b": "refuted"},
        gold_docs={"a": ["d1"], "b": ["d9"]},
        llm=llm,
        base="dense",
        k=2,
    )
    by = {i["claim_id"]: i for i in items}
    assert complete and by["a"]["predicted"] == "supported" and by["a"]["evidence_hit"]
    assert by["b"]["status"] == "failed" and by["b"]["error"] == "bad_output"
    assert not by["b"]["evidence_hit"] and by["b"]["predicted"] == "failed"


async def test_failed_retrieval_is_failed_not_unverified(deps):
    llm = FakeLLM(lambda m: '{"label": "supported", "reason": "r"}')
    items, complete = await run_claims(
        deps=deps,
        collection="missing",
        claims={"a": "solar"},
        labels={"a": "unverified"},
        gold_docs={"a": []},
        llm=llm,
        base="dense",
    )
    assert complete and llm.calls == []
    assert (items[0]["status"], items[0]["predicted"]) == ("failed", "failed")


async def test_budget_stop_marks_incomplete(deps):
    chat = mock_chat(TokenBudget(150))
    items, complete = await run_claims(
        deps=deps,
        collection="c",
        claims={"a": "solar", "b": "wind", "c": "bread"},
        labels={"a": "supported", "b": "supported", "c": "supported"},
        gold_docs={"a": ["d1"], "b": ["d2"], "c": ["d3"]},
        llm=chat,
        base="dense",
        concurrency=1,
    )
    assert not complete and [i["claim_id"] for i in items] == ["a", "b"]
    assert items[0]["tokens"] == 100


async def test_auth_error_raises_and_keeps_no_items(deps):
    with pytest.raises(AuthFailed):
        await run_claims(
            deps=deps,
            collection="c",
            claims={"a": "solar", "b": "wind"},
            labels={"a": "supported", "b": "supported"},
            gold_docs={"a": ["d1"], "b": ["d2"]},
            llm=mock_chat(status=401),
            base="dense",
            concurrency=1,
        )


async def test_results_never_hold_claim_text(deps):
    llm = FakeLLM(lambda m: '{"label": "supported", "reason": "secret reason words"}')
    items, _ = await run_claims(
        deps=deps,
        collection="c",
        claims={"a": "solar panels on roofs"},
        labels={"a": "supported"},
        gold_docs={"a": ["d1"]},
        llm=llm,
        base="dense",
        k=2,
    )
    assert set(items[0]) == ITEM_KEYS
    written = json.dumps(
        {"meta": META, "summary": summarize_verdicts(items), "per_item": items}, indent=1
    )
    assert "solar panels on roofs" not in written and "secret reason words" not in written
    assert '"reason"' not in written and '"claim"' not in written


def test_parse_args():
    with pytest.raises(SystemExit):
        parse_args([])  # --max-tokens required
    with pytest.raises(SystemExit):
        parse_args(["--estimate-only", "--base", "nope"])
    assert parse_args(["--estimate-only"]).sample == 300
    assert parse_args(["--compare", "a.json", "b.json"]).compare == ["a.json", "b.json"]


def test_preflight_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        preflight(parse_args(["--max-tokens", "10"]))


def test_preflight_needs_no_key_for_estimate_or_compare(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert preflight(parse_args(["--estimate-only"])) is None
    assert preflight(parse_args(["--compare", "a.json", "b.json"])) is None


def test_preflight_builds_a_capped_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = preflight(parse_args(["--max-tokens", "10"]))
    assert llm.budget.limit == 10


def test_estimate_claim_counts_prompt_and_cap():
    assert estimate_claim("c" * 30, [text_row("x" * 300, title="t")]) >= 256 + (300 + 30) // 3


def test_report_md_tables():
    md = report_md({"meta": META, "summary": summarize_verdicts(ITEMS)})
    assert "| supported | 2 | 2 | 1 | 0.500 | 0.500 |" in md
    assert "| unverified | 2 | 2 | 1 | 0.500 | 1.000 |" in md
    assert "| Gold \\ predicted | supported | refuted | unverified | failed |" in md
    assert "| unverified | 0 | 0 | 1 | 1 |" in md
    assert "Accuracy 0.600 on 5 claims, 1 failed, unverified 40.0%" in md
    assert "gold evidence in top 5: 75.0% (accuracy 0.667 when found, 0.000 when not)." in md
    assert "5,183 abstracts" in md and "one run" in md and "excluded from accuracy" in md
    assert "Incomplete" not in md
    assert "Incomplete" in report_md(
        {"meta": {**META, "complete": False}, "summary": summarize_verdicts(ITEMS)}
    )


def test_compare_writes_runs_file_without_key_or_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    run = {"meta": META, "summary": summarize_verdicts(ITEMS), "per_item": ITEMS}
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(run))
    assert asyncio.run(main(["--compare", "a.json", "b.json"])) == 0
    text = (tmp_path / "results" / "fact_check_scifact_runs.md").read_text()
    assert "Same label on 100.0% of 5 claims" in text and text == compare_md(run, run)


def test_auth_failure_writes_no_file(tmp_path, monkeypatch, deps):
    """The CLI exits with a status-only message and writes nothing."""
    from production_rag.core.eval.datasets import ClaimDataset

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test")
    monkeypatch.setattr(fc_eval, "build_deps", lambda settings: deps)
    monkeypatch.setattr(
        fc_eval,
        "load_claims",
        lambda sample: ClaimDataset({"a": "solar"}, {"a": "supported"}, {"a": ["d1"]}),
    )
    monkeypatch.setattr(
        fc_eval, "load_dataset", lambda *a, **k: type("D", (), {"corpus": {"d1": (None, "x")}})()
    )

    async def noop(*args, **kwargs):
        return 1

    monkeypatch.setattr(fc_eval, "index_chunks", noop)
    monkeypatch.setattr(fc_eval, "warm_up", noop)
    monkeypatch.setattr(fc_eval, "collection_for", lambda d, m: "c")
    monkeypatch.setattr(
        OpenAIChat, "from_env", classmethod(lambda cls, **kw: mock_chat(kw["budget"], 401))
    )
    with pytest.raises(SystemExit) as exc:
        asyncio.run(main(["--max-tokens", "1000", "--base", "dense"]))
    assert "OPENAI_API_KEY" in str(exc.value) and "sk-secret-test" not in str(exc.value)
    assert not (tmp_path / "results").exists()

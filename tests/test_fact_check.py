import httpx
import pytest
from fakes import FakeLLM, text_row

from production_rag.core.llm import AuthFailed, BudgetExceeded, OpenAIChat, Usage
from production_rag.core.models import SpecResult
from production_rag.methods.fact_check import pipeline
from production_rag.methods.fact_check.claims import Claim, claim_messages, extract_claims
from production_rag.methods.fact_check.pipeline import aggregate, check_claim, fact_check
from production_rag.methods.fact_check.verdict import LABELS, judge_claim, verdict_messages


def refused_chat(status: int) -> OpenAIChat:
    body = {"error": {"code": "invalid_api_key", "message": "bad key sk-test"}}
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    return OpenAIChat("sk-test", client=httpx.AsyncClient(transport=transport), backoff_s=0)


def test_verdict_messages_carry_claim_and_every_passage():
    rows = [
        text_row("tea came from China", title="History of tea"),
        text_row("ceremony uses matcha"),
    ]
    sys, user = (m["content"] for m in verdict_messages("Tea came from Korea", rows))
    assert all(lbl in sys for lbl in LABELS)
    assert "Tea came from Korea" in user and "History of tea" in user
    assert (
        user.index("[1]")
        < user.index("History of tea")
        < user.index("tea came from China")
        < user.index("[2]")
        < user.index("matcha")
    )


def test_claim_messages_carry_statement_and_cap():
    sys, user = (m["content"] for m in claim_messages("Solar is cheap and tea is old.", 3))
    assert user == "Solar is cheap and tea is old." and "3" in sys


async def test_extract_claims_sends_statement_and_cap():
    llm = FakeLLM(lambda m: '{"claims": []}')
    await extract_claims(llm, "Solar is cheap.", max_claims=2)
    sys, user = (m["content"] for m in llm.calls[0])
    assert user == "Solar is cheap." and "2" in sys


async def test_check_claim_sends_retrieved_rows_to_the_verdict(deps):
    llm = FakeLLM(lambda m: '{"label": "supported", "reason": "r"}')
    out = await check_claim("solar panels on roofs", "c", deps, llm, base="dense", k=2)
    sent = llm.calls[0][-1]["content"]
    assert out.label == "supported" and len(out.rows) == 2
    assert "solar panels on roofs" in sent and all(r.text in sent for r in out.rows)
    label = llm.schemas[0]["schema"]["properties"]["label"]
    assert tuple(label["enum"]) == LABELS


async def test_retrieval_failure_skips_the_llm(deps, monkeypatch):
    async def failed(*args, **kwargs):
        return SpecResult(status="failed", elapsed_ms=0, error="boom")

    monkeypatch.setattr(pipeline, "search", failed)
    llm = FakeLLM(lambda m: "x")
    out = await check_claim("solar panels on roofs", "c", deps, llm)
    assert out.label is None and out.error == "retrieval_failed" and llm.calls == []


async def test_check_claim_searches_with_the_query(deps):
    llm = FakeLLM(lambda m: '{"label": "supported", "reason": "r"}')
    out = await check_claim("a claim", "c", deps, llm, query="solar panels", base="bm25", k=1)
    assert out.query == "solar panels" and out.rows[0].doc_id == "d1"
    assert "Claim: a claim" in llm.calls[0][-1]["content"]


@pytest.mark.parametrize("reply", ["nope", '{"label": "true", "reason": ""}'])
async def test_bad_verdict_is_failed_not_unverified(reply):
    v = await judge_claim(FakeLLM(lambda m: reply), "x", [text_row("p")])
    assert v.label is None and v.error == "bad_output"


async def test_truncated_verdict_is_failed():
    llm = FakeLLM(lambda m: '{"label": "supported"', finish_reason="length")
    assert (await judge_claim(llm, "x", [text_row("p")])).error == "bad_output"


async def test_failed_call_is_llm_error():
    class Down:
        async def complete(self, *a, **k):
            raise RuntimeError("boom")

    v = await judge_claim(Down(), "x", [text_row("p")])
    assert v.label is None and v.error == "llm_error"
    assert (await extract_claims(Down(), "s")).error == "llm_error"


async def test_no_rows_is_unverified_without_call():
    llm = FakeLLM(lambda m: "x")
    assert (await judge_claim(llm, "x", [])).label == "unverified" and llm.calls == []


@pytest.mark.parametrize("reply", ["nope", '{"claims": "x"}', '{"claims": [{"query": "q"}]}'])
async def test_bad_extraction_is_bad_output(reply):
    ex = await extract_claims(FakeLLM(lambda m: reply), "s")
    assert ex.claims == [] and ex.error == "bad_output"


async def test_truncated_extraction_is_bad_output():
    llm = FakeLLM(lambda m: '{"claims": []}', finish_reason="length")
    assert (await extract_claims(llm, "s")).error == "bad_output"


async def test_extract_claims_dedupes_and_caps():
    reply = (
        '{"claims": [{"claim": "A", "query": ""}, {"claim": "a", "query": "q"}, '
        '{"claim": " ", "query": "q"}, {"claim": "B", "query": "b"}]}'
    )
    ex = await extract_claims(FakeLLM(lambda m: reply), "s", max_claims=1)
    assert ex.claims == [Claim("A", "A")] and ex.error is None


def test_aggregate():
    assert aggregate([]) == "not_relevant"
    assert aggregate(["supported", "refuted", "unverified"]) == "refuted"
    assert aggregate(["supported", "unverified"]) == "unverified"
    assert aggregate(["supported"]) == "supported"


def scripted_llm() -> FakeLLM:
    def reply(messages):
        if "checkable factual claims" in messages[0]["content"]:
            return (
                '{"claims": [{"claim": "solar panels on roofs", "query": "solar panels"},'
                ' {"claim": "tea farming today", "query": "tea farming"}]}'
            )
        label = "refuted" if "tea" in messages[1]["content"].split("\n")[0] else "supported"
        return f'{{"label": "{label}", "reason": "r"}}'

    return FakeLLM(reply)


async def test_fact_check_end_to_end(deps):
    llm = scripted_llm()
    statement = "Solar panels sit on roofs and tea is farmed today."
    out = await fact_check(statement, "c", deps, llm, base="dense")
    assert out.label == "refuted" and [c.label for c in out.claims] == ["supported", "refuted"]
    assert out.usage == Usage(30, 15)
    assert llm.calls[0][-1]["content"] == statement


async def test_fact_check_failed_claim_counts_as_unverified(deps):
    def reply(messages):
        if "checkable factual claims" in messages[0]["content"]:
            return '{"claims": [{"claim": "solar panels on roofs", "query": ""}]}'
        return "nope"

    out = await fact_check("s", "c", deps, FakeLLM(reply), base="dense")
    assert out.label == "unverified"
    assert out.claims[0].label is None and out.claims[0].error == "bad_output"


async def test_fact_check_extraction_failure(deps):
    out = await fact_check("s", "c", deps, FakeLLM(lambda m: "nope"))
    assert out.label is None and out.error == "extraction_failed" and out.claims == []


async def test_fact_check_no_claims_is_not_relevant(deps):
    out = await fact_check("Hello there!", "c", deps, FakeLLM(lambda m: '{"claims": []}'))
    assert out.label == "not_relevant" and out.claims == []


class Broke:
    async def complete(self, *a, **k):
        raise BudgetExceeded("x")


async def test_budget_exceeded_propagates():
    with pytest.raises(BudgetExceeded):
        await judge_claim(Broke(), "x", [text_row("p")])
    with pytest.raises(BudgetExceeded):
        await extract_claims(Broke(), "s")


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_raises_auth_failed(deps, status):
    chat = refused_chat(status)
    with pytest.raises(AuthFailed) as exc:
        await judge_claim(chat, "x", [text_row("p")])
    assert str(status) in str(exc.value) and "sk-test" not in str(exc.value)
    with pytest.raises(AuthFailed):
        await extract_claims(chat, "s")
    with pytest.raises(AuthFailed):
        await check_claim("solar panels on roofs", "c", deps, chat, base="dense")
    with pytest.raises(AuthFailed):
        await fact_check("s", "c", deps, chat, base="dense")


async def test_other_http_error_is_a_failed_verdict():
    v = await judge_claim(refused_chat(400), "x", [text_row("p")])
    assert v.label is None and v.error == "llm_error"

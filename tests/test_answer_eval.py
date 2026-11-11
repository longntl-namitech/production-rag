import dataclasses
import json

import httpx
import pytest
from fakes import FakeLLM, text_row

from production_rag.core.eval import answer
from production_rag.core.eval.answer import (
    ANSWER_METHODS,
    FRONT_ENDS,
    collection_for,
    estimate_item,
    estimate_tokens,
    parse_args,
    preflight,
    run_items,
    summarize_answers,
)
from production_rag.core.eval.datasets import AnswerDataset
from production_rag.core.llm import OpenAIChat, TokenBudget


class FakeReranker:
    async def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(set(query.split()) & set(t.split()))) for t in texts]


def mock_chat(budget: TokenBudget | None = None, status: int = 200) -> OpenAIChat:
    """Every call costs 100 tokens; judge calls get the first label of their schema.

    With a non-200 status every call fails with that status and an OpenAI-style error body.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            error = {"code": "invalid_api_key", "message": "Incorrect API key provided: sk-te**st"}
            return httpx.Response(status, json={"error": error})
        body = json.loads(request.content)
        fmt = body.get("response_format")
        if fmt is None:
            content = "solar panels"
        else:
            label = fmt["json_schema"]["schema"]["properties"]["label"]["enum"][0]
            content = json.dumps({"label": label, "reason": "r"})
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


def judged(reply_for_answer: str = "solar panels"):
    """Fake replies: a verdict for judge prompts, an answer otherwise."""

    def reply(messages):
        system = messages[0]["content"]
        if "faithful" in system:
            return '{"label": "faithful", "reason": "r"}'
        if "partially_correct" in system:
            return '{"label": "correct", "reason": "r"}'
        return reply_for_answer

    return reply


async def test_abstention_skips_judges(deps):
    llm = FakeLLM(lambda m: "NO_ANSWER")
    items, complete = await run_items(
        deps=deps,
        collection="c",
        methods=["naive_rag"],
        questions={"q1": "solar"},
        gold={"q1": "g"},
        llm=llm,
    )
    assert complete and len(llm.calls) == 1
    item = items[0]
    assert (item["correctness"], item["faithfulness"], item["abstained"]) == (
        "incorrect",
        "faithful",
        True,
    )
    assert item["status"] == "ok" and item["tokens"]["judge"] == 0


async def test_answered_item_is_judged_and_keeps_no_text(deps):
    llm = FakeLLM(judged())
    items, complete = await run_items(
        deps=deps,
        collection="c",
        methods=["naive_rag"],
        questions={"q1": "solar panels"},
        gold={"q1": "gold answer"},
        llm=llm,
        k=2,
    )
    item = items[0]
    assert complete and len(llm.calls) == 3
    assert (item["status"], item["correctness"], item["faithfulness"]) == (
        "ok",
        "correct",
        "faithful",
    )
    assert item["answer"] == "solar panels" and item["doc_ids"]
    assert item["tokens"] == {"rag": 15, "judge": 30, "reasoning": 0}
    dumped = json.dumps(item)
    assert "gold answer" not in dumped and '"reason"' not in dumped
    faith_prompt = next(c for c in llm.calls if "faithful" in c[0]["content"])[-1]["content"]
    assert "Question: solar panels" in faith_prompt


async def test_judge_bad_output_marks_judge_failed(deps):
    def reply(messages):
        return "not json" if "partially_correct" in messages[0]["content"] else judged()(messages)

    items, _ = await run_items(
        deps=deps,
        collection="c",
        methods=["naive_rag"],
        questions={"q1": "solar"},
        gold={"q1": "g"},
        llm=FakeLLM(reply),
    )
    item = items[0]
    assert (item["status"], item["correctness"], item["faithfulness"]) == (
        "judge_failed",
        None,
        "faithful",
    )
    assert item["correctness_error"] == "bad_output"


async def test_failed_retrieval_is_failed_not_abstained(deps):
    llm = FakeLLM(judged())
    items, complete = await run_items(
        deps=deps,
        collection="missing",
        methods=["naive_rag"],
        questions={"q1": "solar"},
        gold={"q1": "g"},
        llm=llm,
    )
    assert complete and llm.calls == []
    assert (items[0]["status"], items[0]["abstained"], items[0]["correctness"]) == (
        "failed",
        False,
        None,
    )


async def test_budget_stop_keeps_items_and_marks_incomplete(deps):
    chat = mock_chat(TokenBudget(250))
    items, complete = await run_items(
        deps=deps,
        collection="c",
        methods=["naive_rag"],
        questions={"q1": "solar", "q2": "wind"},
        gold={"q1": "g", "q2": "g"},
        llm=chat,
        concurrency=1,
    )
    assert not complete and [i["qid"] for i in items] == ["q1"]
    assert chat.budget.used == 300


def _item(status="ok", correctness=None, faithfulness=None, abstained=False):
    return {
        "front_end": "naive_rag",
        "status": status,
        "abstained": abstained,
        "correctness": correctness,
        "faithfulness": faithfulness,
        "retrieval_ms": 10,
        "rewrite_ms": None,
        "gen_ms": None if status == "failed" or abstained else 100,
        "passage_tokens": None if status == "failed" else 50,
        "tokens": {"rag": 30, "judge": 60, "reasoning": 0},
    }


def test_summarize_answers():
    items = [
        _item(correctness="correct", faithfulness="faithful"),
        _item(correctness="partially_correct", faithfulness="unfaithful"),
        _item(correctness="incorrect", faithfulness="faithful", abstained=True),
        _item(status="failed"),
        _item(status="judge_failed", faithfulness="faithful"),
    ]
    s = summarize_answers(items, ["naive_rag"])["naive_rag"]
    assert (s["questions"], s["failed"], s["judge_failed"]) == (5, 1, 1)
    assert s["correctness"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert s["faithfulness"] == pytest.approx(3 / 4) and s["abstained"] == pytest.approx(1 / 4)
    assert (s["retrieval_p50_ms"], s["generation_p50_ms"]) == (10, 100)
    assert (s["rag_tokens"], s["judge_tokens"], s["passage_tokens"]) == (30, 60, 50)
    assert "rewrite_p50_ms" not in s


def test_summarize_multi_query_has_rewrite_latency():
    item = {**_item(correctness="correct", faithfulness="faithful"), "front_end": "multi_query"}
    item["rewrite_ms"] = 40
    assert summarize_answers([item], ["multi_query"])["multi_query"]["rewrite_p50_ms"] == 40


def test_parse_args_rejects_unknown_method(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "mini_wiki", "--methods", "dense", "--max-tokens", "1"])
    assert "naive_rag" in capsys.readouterr().err


def test_parse_args_needs_max_tokens_for_llm_runs():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "vi_legal"])
    args = parse_args(["--dataset", "vi_legal", "--estimate-only"])
    assert (args.sample, args.k, args.concurrency, args.methods) == (
        100,
        5,
        4,
        list(ANSWER_METHODS),
    )


def test_preflight_without_key_exits_with_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        preflight(parse_args(["--dataset", "mini_wiki", "--max-tokens", "10"]))
    assert preflight(parse_args(["--dataset", "mini_wiki", "--estimate-only"])) is None


def test_preflight_sets_the_budget(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chat = preflight(parse_args(["--dataset", "mini_wiki", "--max-tokens", "10"]))
    assert chat.budget.limit == 10


def test_estimate_item_counts_prompts_and_caps():
    rows = [text_row("x" * 300)]
    base = estimate_item("q" * 30, "g" * 30, rows, "naive_rag")
    assert estimate_item("q" * 30, "g" * 30, rows, "multi_query") > base
    assert base >= 512 + 2 * (256 + 512) + estimate_tokens("x" * 300) * 2
    assert estimate_tokens("abcd") == 2


def test_collection_for():
    assert collection_for("mini_wiki") == "answer-mini_wiki-c800"


async def test_front_ends_all_return_rows(deps):
    deps = dataclasses.replace(deps, reranker=FakeReranker())
    llm = FakeLLM(lambda m: '{"queries": ["wind turbines"]}')
    for name in ANSWER_METHODS:
        got = await FRONT_ENDS[name]("solar panels", "c", deps, llm, 2)
        assert got.rows and got.status == "ok", name
    assert len(llm.calls) == 1


async def test_multi_query_without_llm_searches_the_question_only(deps):
    got = await FRONT_ENDS["multi_query"]("solar panels", "c", deps, None, 2)
    assert got.rows and got.rewrite_ms is None and got.usage.total == 0


async def test_rerank_front_end_refuses_without_reranker(deps):
    with pytest.raises(RuntimeError, match="local-models"):
        await FRONT_ENDS["rerank"]("solar", "c", deps, None, 2)


@pytest.fixture
def cli(deps, monkeypatch, tmp_path):
    ds = AnswerDataset(
        "mini_wiki",
        corpus={"w1": (None, "solar panels on roofs"), "w2": (None, "wind turbines offshore")},
        questions={"q1": "solar panels", "q2": "wind turbines"},
        gold={"q1": "roof panels", "q2": "offshore"},
    )
    with_reranker = dataclasses.replace(deps, reranker=FakeReranker())
    monkeypatch.setattr(answer, "build_deps", lambda settings: with_reranker)
    monkeypatch.setattr(answer, "load_answer_dataset", lambda name, sample: ds)
    monkeypatch.setattr(answer, "RESULTS", tmp_path)
    return tmp_path


async def test_cli_budget_stop_writes_incomplete_and_exits_nonzero(cli, monkeypatch, capsys):
    monkeypatch.setattr(answer, "preflight", lambda args: mock_chat(TokenBudget(args.max_tokens)))
    code = await answer.main(
        ["--dataset", "mini_wiki", "--methods", "naive_rag", "--max-tokens", "250"]
        + ["--concurrency", "1"]
    )
    assert code == 1
    run = json.loads((cli / "answer_mini_wiki.json").read_text())
    assert run["meta"]["complete"] is False and run["meta"]["kind"] == "answer"
    assert [i["qid"] for i in run["per_item"]] == ["q1"]
    assert run["meta"]["tokens"] == {
        "total": 300,
        "rag": 100,
        "judge": 200,
        "dropped": 0,
        "reasoning": 0,
    }
    text = json.dumps(run)
    for secret in ["roof panels", "solar panels on roofs", "sk-test", "reasons"]:
        assert secret not in text
    assert "tokens: rag=" in capsys.readouterr().out


async def test_cli_estimate_only_makes_no_llm_calls(cli, monkeypatch, capsys):
    fake = FakeLLM(lambda m: "x")
    monkeypatch.setattr(answer, "preflight", lambda args: fake)
    code = await answer.main(["--dataset", "mini_wiki", "--estimate-only", "--sample", "2"])
    assert code == 0 and fake.calls == []
    out = capsys.readouterr().out
    assert "estimate: questions=2 methods=5 llm_calls=32 upper_bound_tokens=" in out
    assert list(cli.iterdir()) == []


async def test_cli_rerank_without_reranker_stops_before_indexing(cli, deps, monkeypatch):
    monkeypatch.setattr(answer, "build_deps", lambda settings: deps)
    monkeypatch.setattr(answer, "preflight", lambda args: None)
    monkeypatch.setattr(answer, "load_answer_dataset", pytest.fail)
    with pytest.raises(SystemExit, match="local-models"):
        await answer.main(["--dataset", "mini_wiki", "--methods", "rerank", "--estimate-only"])


async def test_cli_without_key_stops_before_deps_and_download(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(answer, "build_deps", pytest.fail)
    monkeypatch.setattr(answer, "load_answer_dataset", pytest.fail)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        await answer.main(["--dataset", "vi_legal", "--max-tokens", "10"])


class BrokenReranker:
    async def score(self, query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("remote reranker down")


async def test_rerank_not_applied_is_failed_not_scored(deps):
    deps = dataclasses.replace(deps, reranker=BrokenReranker())
    llm = FakeLLM(judged())
    items, complete = await run_items(
        deps=deps,
        collection="c",
        methods=["rerank"],
        questions={"q1": "solar panels"},
        gold={"q1": "g"},
        llm=llm,
    )
    assert complete and llm.calls == []
    item = items[0]
    assert (item["status"], item["error"], item["rerank"]) == ("failed", "rerank failed", "failed")


async def test_reranked_item_records_rerank_status(deps):
    deps = dataclasses.replace(deps, reranker=FakeReranker())
    items, _ = await run_items(
        deps=deps,
        collection="c",
        methods=["rerank"],
        questions={"q1": "solar panels"},
        gold={"q1": "g"},
        llm=FakeLLM(judged()),
    )
    assert (items[0]["status"], items[0]["rerank"]) == ("ok", "ok")


async def test_unexpected_error_fails_one_question_and_the_run_continues(deps, monkeypatch):
    real = answer.judge_correctness

    async def flaky(llm, question, gold, text):
        if question == "solar":
            raise TypeError("bug")
        return await real(llm, question, gold, text)

    monkeypatch.setattr(answer, "judge_correctness", flaky)
    items, complete = await run_items(
        deps=deps,
        collection="c",
        methods=["naive_rag"],
        questions={"q1": "solar", "q2": "wind"},
        gold={"q1": "g", "q2": "g"},
        llm=FakeLLM(judged()),
    )
    assert complete
    assert [(i["qid"], i["status"], i["error"]) for i in items] == [
        ("q1", "failed", "TypeError"),
        ("q2", "ok", None),
    ]


async def test_cli_rejected_key_is_fatal_and_writes_nothing(cli, monkeypatch):
    monkeypatch.setattr(answer, "preflight", lambda args: mock_chat(status=401))
    with pytest.raises(SystemExit, match="OPENAI_API_KEY") as exc:
        await answer.main(["--dataset", "mini_wiki", "--methods", "naive_rag", "--max-tokens", "9"])
    assert "sk-te" not in str(exc.value) and "401" in str(exc.value)
    assert list(cli.iterdir()) == []


async def test_cli_every_generation_failed_is_incomplete(cli, monkeypatch, capsys):
    monkeypatch.setattr(answer, "preflight", lambda args: mock_chat(status=500))
    code = await answer.main(
        ["--dataset", "mini_wiki", "--methods", "naive_rag", "--max-tokens", "9"]
    )
    assert code == 1
    run = json.loads((cli / "answer_mini_wiki.json").read_text())
    assert run["meta"]["complete"] is False
    assert {i["error"] for i in run["per_item"]} == {"generation LLMError"}
    assert "every generation call failed" in capsys.readouterr().out


async def test_cli_budget_stop_counts_tokens_of_dropped_questions(cli, monkeypatch):
    monkeypatch.setattr(answer, "preflight", lambda args: mock_chat(TokenBudget(args.max_tokens)))
    code = await answer.main(
        ["--dataset", "mini_wiki", "--methods", "naive_rag", "--max-tokens", "150"]
        + ["--concurrency", "1"]
    )
    run = json.loads((cli / "answer_mini_wiki.json").read_text())
    assert code == 1 and run["per_item"] == []
    tokens = run["meta"]["tokens"]
    assert (tokens["total"], tokens["rag"], tokens["judge"], tokens["dropped"]) == (200, 0, 0, 200)


async def test_budget_stop_with_concurrency_drops_cut_questions(deps):
    chat = mock_chat(TokenBudget(250))
    items, complete = await run_items(
        deps=deps,
        collection="c",
        methods=["naive_rag"],
        questions={f"q{i}": "solar" for i in range(4)},
        gold={f"q{i}": "g" for i in range(4)},
        llm=chat,
        concurrency=2,
    )
    assert not complete
    assert all(i["status"] == "ok" and i["faithfulness"] for i in items)
    # at most `concurrency` calls pass the cap: 250 + 2 * 100
    assert 300 <= chat.budget.used <= 450

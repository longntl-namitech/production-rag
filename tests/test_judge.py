import pytest
from fakes import FakeLLM, text_row

from production_rag.core.eval.judge import SCORES, judge_correctness, judge_faithfulness
from production_rag.core.llm import BudgetExceeded, LLMError, Usage
from production_rag.methods.multi_query.rewrite import rewrite


class RaisingLLM:
    model = "fake"

    def __init__(self, exc: Exception):
        self.exc = exc

    async def complete(self, messages, *, max_tokens, json_schema=None):
        raise self.exc


async def test_correctness_parses_label():
    llm = FakeLLM(lambda m: '{"label": "partially_correct", "reason": "r"}')
    v = await judge_correctness(llm, "q", "gold text", "answer text")
    assert (v.label, SCORES[v.label], v.error) == ("partially_correct", 0.5, None)
    assert v.usage == Usage(10, 5)
    prompt = llm.calls[0][-1]["content"]
    assert "gold text" in prompt and "answer text" in prompt


async def test_faithfulness_parses_label_and_sends_passages():
    llm = FakeLLM(lambda m: '{"label": "unfaithful", "reason": "r"}')
    v = await judge_faithfulness(llm, "a", [text_row("passage one")])
    assert (v.label, v.error) == ("unfaithful", None)
    assert "passage one" in llm.calls[0][-1]["content"]


@pytest.mark.parametrize(
    "reply", ["not json", '{"label": "maybe", "reason": ""}', '["faithful"]', ""]
)
async def test_bad_judge_output_is_not_a_score(reply):
    v = await judge_faithfulness(FakeLLM(lambda m: reply), "a", [text_row("p")])
    assert v.label is None and v.error == "bad_output" and v.usage == Usage(10, 5)


async def test_truncated_judge_output_is_bad():
    llm = FakeLLM(lambda m: '{"label": "correct", "reason": "r"}', finish_reason="length")
    assert (await judge_correctness(llm, "q", "g", "a")).error == "bad_output"


@pytest.mark.parametrize("exc", [LLMError("openai 500"), KeyError("choices")])
async def test_judge_call_error_is_not_a_score(exc):
    v = await judge_correctness(RaisingLLM(exc), "q", "g", "a")
    assert (v.label, v.error) == (None, "llm_error")


async def test_judge_budget_exceeded_propagates():
    with pytest.raises(BudgetExceeded):
        await judge_faithfulness(RaisingLLM(BudgetExceeded("cap")), "a", [text_row("p")])


async def test_rewrite_returns_distinct_rephrasings():
    qs, usage = await rewrite(
        FakeLLM(lambda m: '{"queries": ["a", "a", "q", "b", " ", "c", "d"]}'), "q", n=3
    )
    assert qs == ["a", "b", "c"] and usage == Usage(10, 5)


@pytest.mark.parametrize("reply", ["nope", '{"queries": "a"}', '{"other": ["a"]}'])
async def test_rewrite_bad_output_falls_back(reply):
    qs, usage = await rewrite(FakeLLM(lambda m: reply), "q")
    assert qs == [] and usage == Usage(10, 5)


async def test_rewrite_truncated_or_error_falls_back():
    llm = FakeLLM(lambda m: '{"queries": ["a"]}', finish_reason="length")
    assert (await rewrite(llm, "q"))[0] == []
    assert await rewrite(RaisingLLM(LLMError("openai 500")), "q") == ([], Usage())
    with pytest.raises(BudgetExceeded):
        await rewrite(RaisingLLM(BudgetExceeded("cap")), "q")

import pytest
from fakes import FakeLLM, text_row

from production_rag.core.generation import NO_ANSWER, build_messages, generate
from production_rag.core.llm import Usage


def test_messages_number_passages_and_end_with_question():
    msgs = build_messages("Why?", [text_row("p one", title="T"), text_row("p two")])
    user = msgs[-1]["content"]
    assert user.index("[1]") < user.index("p one") < user.index("[2]") < user.index("Why?")
    assert "T" in user.split("p one")[0]
    assert "NO_ANSWER" in msgs[0]["content"]


async def test_generate_without_rows_makes_no_call():
    llm = FakeLLM(lambda m: "x")
    out = await generate(llm, "q", [])
    assert out.abstained and out.text == NO_ANSWER and llm.calls == []
    assert out.finish_reason == "no_passages" and out.usage == Usage()


async def test_generate_detects_abstention():
    out = await generate(FakeLLM(lambda m: " NO_ANSWER\n"), "q", [text_row("p")])
    assert out.abstained and out.usage == Usage(10, 5)


async def test_generate_returns_answer():
    out = await generate(FakeLLM(lambda m: "Yes."), "q", [text_row("p")])
    assert not out.abstained and out.text == "Yes." and out.finish_reason == "stop"


@pytest.mark.parametrize(
    "reply",
    ["NO_ANSWER", "[NO_ANSWER]", "NO_ANSWER.", "`NO_ANSWER`", '"NO_ANSWER"', " 'NO_ANSWER' \n"],
)
async def test_generate_detects_wrapped_abstention(reply):
    out = await generate(FakeLLM(lambda m: reply), "q", [text_row("p")])
    assert out.abstained and out.text == reply


@pytest.mark.parametrize(
    "reply",
    [
        "Paris. NO_ANSWER",
        "The passages give NO_ANSWER for this.",
        "NO_ANSWER is not it",
        "NOT_ANSWER",
    ],
)
async def test_answer_containing_token_is_not_abstention(reply):
    out = await generate(FakeLLM(lambda m: reply), "q", [text_row("p")])
    assert not out.abstained

import json

from fakes import FakeLLM

from production_rag.core.llm import Usage
from production_rag.methods import METHOD_NAMES
from production_rag.methods.hyde import pipeline as hyde
from production_rag.methods.hyde.generate import hyde_messages, queries_for

QUESTION = "how do solar cells make power"


async def test_queries_for_returns_passage():
    llm = FakeLLM(lambda m: json.dumps({"passage": "Solar cells turn light into current."}))
    queries, usage = await queries_for(llm, QUESTION)
    assert queries == ["Solar cells turn light into current."]
    assert usage == Usage(10, 5)


async def test_queries_for_with_question_puts_question_first():
    llm = FakeLLM(lambda m: json.dumps({"passage": "Solar cells turn light into current."}))
    queries, _ = await queries_for(llm, QUESTION, include_question=True)
    assert queries == [QUESTION, "Solar cells turn light into current."]


async def test_queries_for_falls_back_to_question_on_bad_reply():
    queries, _ = await queries_for(FakeLLM(lambda m: "nope"), QUESTION)
    assert queries == [QUESTION]


async def test_queries_for_falls_back_on_blank_passage():
    llm = FakeLLM(lambda m: json.dumps({"passage": "   "}))
    queries, _ = await queries_for(llm, QUESTION, include_question=True)
    assert queries == [QUESTION]


def test_hyde_messages_carry_the_question():
    assert QUESTION in hyde_messages(QUESTION)[-1]["content"]


async def test_hyde_search_uses_dense_only(deps):
    result = await hyde.search(["solar panels on roofs"], "c", deps, k=2)
    assert result.rows[0].doc_id == "d1"
    assert deps.bm25.calls == 0


def test_hyde_registered():
    assert "hyde" in METHOD_NAMES
    assert hyde.VARIANTS == {} and hyde.DEFAULT is None

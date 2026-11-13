import json

from fakes import FakeLLM

from production_rag.methods import METHOD_NAMES
from production_rag.methods.multi_query import pipeline as multi_query
from production_rag.methods.step_back import pipeline as step_back
from production_rag.methods.step_back.generate import queries_for, step_back_messages

QUESTION = "why did the 1929 crash hit German banks"


async def test_queries_for_adds_step_back_after_question():
    llm = FakeLLM(lambda m: json.dumps({"question": "What caused the Great Depression?"}))
    queries, usage = await queries_for(llm, QUESTION)
    assert queries == [QUESTION, "What caused the Great Depression?"]
    assert usage.total == 15


async def test_queries_for_falls_back_on_failure():
    queries, _ = await queries_for(FakeLLM(lambda m: "nope"), QUESTION)
    assert queries == [QUESTION]


async def test_queries_for_drops_echo():
    llm = FakeLLM(lambda m: json.dumps({"question": QUESTION.upper()}))
    queries, _ = await queries_for(llm, QUESTION)
    assert queries == [QUESTION]


def test_step_back_messages_carry_the_question():
    assert QUESTION in step_back_messages(QUESTION)[-1]["content"]


async def test_step_back_search_merges_both_queries(deps):
    result = await step_back.search(["solar panels", "bread baking"], "c", deps, k=2)
    assert {r.doc_id for r in result.rows} == {"d1", "d3"}


def test_step_back_registered():
    assert "step_back" in METHOD_NAMES
    assert step_back.VARIANTS == {} and step_back.DEFAULT is None
    assert step_back.search is multi_query.search

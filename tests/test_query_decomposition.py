import json

from fakes import FakeLLM

from production_rag.methods import METHOD_NAMES
from production_rag.methods.multi_query import pipeline as multi_query
from production_rag.methods.query_decomposition import pipeline as query_decomposition
from production_rag.methods.query_decomposition.generate import (
    MAX_SUB,
    decompose_messages,
    queries_for,
)

QUESTION = "who wrote Hamlet and when was the Globe theatre built"


def reply(items):
    return FakeLLM(lambda m: json.dumps({"sub_questions": items}))


async def test_returns_sub_questions_capped_at_n():
    queries, _ = await queries_for(reply([f"part {i}?" for i in range(5)]), QUESTION, n=3)
    assert queries == ["part 0?", "part 1?", "part 2?"]


async def test_default_cap_is_max_sub():
    queries, _ = await queries_for(reply([f"part {i}?" for i in range(10)]), QUESTION)
    assert len(queries) == MAX_SUB


async def test_dedupes_sub_questions():
    items = ["Who wrote Hamlet?", " who wrote hamlet? ", "When?"]
    queries, _ = await queries_for(reply(items), QUESTION)
    assert queries == ["Who wrote Hamlet?", "When?"]


async def test_simple_question_single_sub_question_is_kept():
    queries, _ = await queries_for(reply(["Who wrote Hamlet?"]), "who wrote hamlet")
    assert queries == ["Who wrote Hamlet?"]


async def test_falls_back_to_question_on_empty_list():
    queries, _ = await queries_for(reply([]), QUESTION)
    assert queries == [QUESTION]


async def test_falls_back_on_failure():
    queries, _ = await queries_for(FakeLLM(lambda m: "nope"), QUESTION)
    assert queries == [QUESTION]


def test_decompose_messages_carry_question_and_limit():
    text = " ".join(m["content"] for m in decompose_messages(QUESTION, 3))
    assert QUESTION in text and "3" in text


async def test_search_merges_sub_question_results(deps):
    result = await query_decomposition.search(["solar panels", "bread baking"], "c", deps, k=2)
    assert {r.doc_id for r in result.rows} == {"d1", "d3"}


def test_query_decomposition_registered():
    assert "query_decomposition" in METHOD_NAMES
    assert query_decomposition.VARIANTS == {} and query_decomposition.DEFAULT is None
    assert query_decomposition.search is multi_query.search

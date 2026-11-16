import json

from fakes import FakeLLM, text_row
from test_contextual_headers import Boom, BudgetAfter

from production_rag.core.llm import BudgetExceeded, Usage
from production_rag.methods import METHOD_NAMES
from production_rag.methods.contextual_compression import pipeline as contextual_compression
from production_rag.methods.contextual_compression.compress import compress, compress_messages

QUESTION = "how do solar panels make power"


def rows():
    return [
        text_row(
            "Solar panels turn light into power. They sit on roofs. Wind is different.",
            title="Solar",
            doc_id="d1",
            score=0.9,
        ),
        text_row("Bread rises slowly. Yeast eats sugar.", doc_id="d2", score=0.5),
    ]


def keep(*ids: str) -> FakeLLM:
    return FakeLLM(lambda m: json.dumps({"keep": list(ids)}))


async def test_compress_keeps_chosen_sentences_in_order():
    out, usage = await compress(keep("2.1", "1.2", "1.1"), QUESTION, rows())
    assert [(r.doc_id, r.text, r.score, r.title) for r in out] == [
        ("d1", "Solar panels turn light into power. They sit on roofs.", 0.9, "Solar"),
        ("d2", "Bread rises slowly.", 0.5, None),
    ]
    assert usage == Usage(10, 5)


async def test_compress_drops_passage_without_kept_sentence():
    out, _ = await compress(keep("1.1"), QUESTION, rows())
    assert [(r.doc_id, r.text) for r in out] == [("d1", "Solar panels turn light into power.")]


async def test_compress_out_of_range_ids_ignored():
    out, _ = await compress(keep("1.2", "9.9", "1.99", "x", "0.1"), QUESTION, rows())
    assert [(r.doc_id, r.text) for r in out] == [("d1", "They sit on roofs.")]


async def test_compress_all_ids_invalid_returns_original():
    original = rows()
    out, _ = await compress(keep("9.9", "nope"), QUESTION, original)
    assert out == original


async def test_compress_empty_keep_returns_original():
    original = rows()
    out, _ = await compress(keep(), QUESTION, original)
    assert out == original


async def test_compress_bad_reply_or_failure_returns_original():
    original = rows()
    for llm in (
        FakeLLM(lambda m: "nope"),
        FakeLLM(lambda m: json.dumps({"keep": ["1.1"]}), finish_reason="length"),
        Boom(),
    ):
        out, _ = await compress(llm, QUESTION, original)
        assert out == original


async def test_compress_propagates_budget_exceeded():
    try:
        await compress(BudgetAfter(0), QUESTION, rows())
    except BudgetExceeded:
        return
    raise AssertionError("BudgetExceeded did not propagate")


def test_compress_messages_numbers_sentences():
    text = compress_messages(QUESTION, rows())[-1]["content"]
    assert QUESTION in text
    for marker in ("[1.1]", "[1.3]", "[2.1]", "[2.2]"):
        assert marker in text
    assert "[2.3]" not in text


async def test_search_compresses_retrieved_rows(deps):
    result, usage = await contextual_compression.search("solar panels", "c", deps, keep("1.1"), k=2)
    assert [r.doc_id for r in result.rows] == ["d1"] and usage == Usage(10, 5)


def test_contextual_compression_registered():
    assert "contextual_compression" in METHOD_NAMES
    assert contextual_compression.VARIANTS == {} and contextual_compression.DEFAULT is None

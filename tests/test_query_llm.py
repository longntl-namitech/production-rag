import json

import pytest
from fakes import FakeLLM

from production_rag.core.llm import BudgetExceeded, LLMError, Usage
from production_rag.core.query_llm import MAX_QUERY_CHARS, ask, merge_queries

SCHEMA = {"name": "x", "schema": {"type": "object"}}
MESSAGES = [{"role": "user", "content": "q"}]


class Raises:
    def __init__(self, exc: Exception):
        self.exc = exc

    async def complete(self, messages, *, max_tokens, json_schema=None):
        raise self.exc


def parse_a(data):
    return data["a"] if isinstance(data, dict) and "a" in data else None


async def test_ask_returns_parsed_value_and_usage():
    llm = FakeLLM(lambda m: json.dumps({"a": 1}))
    got = await ask(llm, MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a)
    assert got == (1, Usage(10, 5))


async def test_ask_none_on_non_json():
    llm = FakeLLM(lambda m: "nope")
    value, usage = await ask(llm, MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a)
    assert value is None and usage == Usage(10, 5)


async def test_ask_none_on_length_finish():
    llm = FakeLLM(lambda m: json.dumps({"a": 1}), finish_reason="length")
    value, usage = await ask(llm, MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a)
    assert value is None and usage == Usage(10, 5)


async def test_ask_none_when_parse_returns_none():
    llm = FakeLLM(lambda m: json.dumps({"b": 1}))
    value, _ = await ask(llm, MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a)
    assert value is None


async def test_ask_none_on_llm_error():
    value, usage = await ask(
        Raises(LLMError("x")), MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a
    )
    assert value is None and usage == Usage()


async def test_ask_propagates_budget_exceeded():
    with pytest.raises(BudgetExceeded):
        await ask(
            Raises(BudgetExceeded("full")), MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a
        )


async def test_ask_passes_schema_and_cap():
    llm = FakeLLM(lambda m: json.dumps({"a": 1}))
    await ask(llm, MESSAGES, schema=SCHEMA, max_tokens=10, parse=parse_a)
    assert llm.schemas[0] == SCHEMA and llm.calls[0] == MESSAGES


def test_merge_queries_keeps_question_first_and_dedupes():
    assert merge_queries("Q", ["q", " a ", "A", "b"]) == ["Q", "a", "b"]


def test_merge_queries_caps_count_and_length():
    assert len(merge_queries("Q", [f"x{i}" for i in range(20)], limit=8)) == 8
    long = merge_queries("Q", ["y" * 3000])
    assert long[1] == "y" * MAX_QUERY_CHARS

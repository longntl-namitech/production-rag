import json
import os
import subprocess
import sys

import httpx
import pytest
from fakes import FakeLLM

from production_rag.core.llm import (
    BudgetExceeded,
    LLMError,
    MissingAPIKey,
    OpenAIChat,
    TokenBudget,
    Usage,
)

MSGS = [{"role": "user", "content": "q"}]


def ok_body(content: str = "hi") -> dict:
    return {
        "model": "gpt-5.4-nano",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


def make_chat(responses: list[httpx.Response], **kwargs) -> tuple[OpenAIChat, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("backoff_s", 0)
    return OpenAIChat("sk-test", client=client, **kwargs), seen


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey, match="OPENAI_API_KEY"):
        OpenAIChat.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(MissingAPIKey, match="OPENAI_API_KEY"):
        OpenAIChat.from_env()


def test_from_env_model_default_and_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert OpenAIChat.from_env().model == "gpt-5.4-nano"
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
    assert OpenAIChat.from_env().model == "gpt-x"


async def test_complete_sends_request_and_parses_usage():
    chat, seen = make_chat([httpx.Response(200, json=ok_body())])
    out = await chat.complete(MSGS, max_tokens=64)
    req = seen[0]
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["max_completion_tokens"] == 64 and body["reasoning_effort"] == "none"
    assert "response_format" not in body
    assert (out.text, out.usage, out.usage.total) == ("hi", Usage(10, 5, 2), 15)
    assert (out.finish_reason, out.model) == ("stop", "gpt-5.4-nano")


async def test_reasoning_effort_omitted_when_none():
    chat, seen = make_chat([httpx.Response(200, json=ok_body())], reasoning_effort=None)
    await chat.complete(MSGS, max_tokens=8)
    assert "reasoning_effort" not in json.loads(seen[0].content)


async def test_missing_usage_fields_count_as_zero():
    body = ok_body()
    body["usage"] = {}
    chat, _ = make_chat([httpx.Response(200, json=body)])
    assert (await chat.complete(MSGS, max_tokens=8)).usage == Usage(0, 0, 0)


async def test_json_schema_sets_response_format():
    chat, seen = make_chat([httpx.Response(200, json=ok_body())])
    schema = {"name": "v", "schema": {"type": "object"}}
    await chat.complete(MSGS, max_tokens=8, json_schema=schema)
    rf = json.loads(seen[0].content)["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "v"


async def test_retries_429_then_succeeds():
    chat, seen = make_chat([httpx.Response(429), httpx.Response(200, json=ok_body())])
    assert (await chat.complete(MSGS, max_tokens=8)).text == "hi" and len(seen) == 2


async def test_retries_transport_error():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=ok_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = OpenAIChat("sk-test", client=client, backoff_s=0)
    assert (await chat.complete(MSGS, max_tokens=8)).text == "hi" and len(calls) == 2


async def test_gives_up_after_retries():
    chat, seen = make_chat([httpx.Response(503)], max_retries=2)
    with pytest.raises(LLMError, match="503"):
        await chat.complete(MSGS, max_tokens=8)
    assert len(seen) == 3


async def test_401_not_retried_and_key_not_leaked():
    body = {"error": {"message": "Incorrect API key provided: sk-test", "code": "invalid_api_key"}}
    chat, seen = make_chat([httpx.Response(401, json=body)])
    with pytest.raises(LLMError) as exc:
        await chat.complete(MSGS, max_tokens=8)
    assert len(seen) == 1
    assert "sk-test" not in str(exc.value) and "invalid_api_key" in str(exc.value)
    assert "sk-test" not in repr(chat)


async def test_budget_refuses_call_at_limit():
    chat, seen = make_chat([httpx.Response(200, json=ok_body())], budget=TokenBudget(15))
    await chat.complete(MSGS, max_tokens=8)
    with pytest.raises(BudgetExceeded):
        await chat.complete(MSGS, max_tokens=8)
    assert len(seen) == 1 and chat.budget.used == 15


def test_usage_add():
    assert Usage(1, 2, 3) + Usage(10, 20, 30) == Usage(11, 22, 33)


def test_llm_modules_import_without_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    code = "import production_rag.core.llm"
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


async def test_fake_llm_records_calls():
    fake = FakeLLM(lambda m: "x", finish_reason="length")
    out = await fake.complete(MSGS, max_tokens=8)
    assert (out.text, out.usage, out.finish_reason, out.model) == (
        "x",
        Usage(10, 5),
        "length",
        "fake",
    )
    assert fake.calls == [MSGS]

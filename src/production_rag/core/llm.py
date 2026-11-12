"""Minimal OpenAI chat client: retries, token budget, no secrets in errors."""

import asyncio
import os
from dataclasses import dataclass

import httpx

DEFAULT_MODEL = "gpt-5.4-nano"
API_URL = "https://api.openai.com/v1/chat/completions"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class MissingAPIKey(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class AuthFailed(RuntimeError):
    """OpenAI refused the key: every further call would fail the same way."""


def is_auth_error(exc: Exception) -> bool:
    return isinstance(exc, LLMError) and str(exc).startswith(("openai 401", "openai 403"))


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    finish_reason: str
    model: str


class TokenBudget:
    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0

    def check(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise BudgetExceeded(f"token budget reached ({self.used}/{self.limit})")

    def add(self, usage: Usage) -> None:
        self.used += usage.total


def _parse_usage(data: dict) -> Usage:
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return Usage(
        usage.get("prompt_tokens") or 0,
        usage.get("completion_tokens") or 0,
        details.get("reasoning_tokens") or 0,
    )


def _error_label(response: httpx.Response) -> str:
    # Only the error code or type is kept; OpenAI echoes a masked key in the message.
    try:
        error = response.json().get("error") or {}
        label = error.get("code") or error.get("type")
    except (ValueError, AttributeError):
        label = None
    return f"openai {response.status_code} {label}" if label else f"openai {response.status_code}"


class OpenAIChat:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        reasoning_effort: str | None = "none",
        client: httpx.AsyncClient | None = None,
        budget: TokenBudget | None = None,
        max_retries: int = 3,
        backoff_s: float = 1.0,
    ):
        self._api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.budget = budget
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._client = client or httpx.AsyncClient(timeout=120)

    @classmethod
    def from_env(cls, **kwargs) -> "OpenAIChat":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise MissingAPIKey("set OPENAI_API_KEY to run LLM steps")
        return cls(key, os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL, **kwargs)

    def __repr__(self) -> str:
        return f"OpenAIChat(model={self.model!r})"

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> Completion:
        if self.budget is not None:
            self.budget.check()
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {**json_schema, "strict": True},
            }
        data = await self._post(body)
        choice = data["choices"][0]
        usage = _parse_usage(data)
        if self.budget is not None:
            self.budget.add(usage)
        return Completion(
            choice["message"].get("content") or "",
            usage,
            choice.get("finish_reason") or "",
            data.get("model") or self.model,
        )

    async def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = await self._client.post(API_URL, json=body, headers=headers)
            except httpx.TransportError as exc:
                if last:
                    raise LLMError(type(exc).__name__) from None
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in RETRY_STATUSES or last:
                    raise LLMError(_error_label(response))
            await asyncio.sleep(self.backoff_s * 2**attempt)
        raise LLMError("unreachable")  # pragma: no cover

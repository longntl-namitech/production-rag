"""One LLM call for a query-side method, with the fallback rules of `multi_query/rewrite.py`.

Any output that cannot be used (not JSON, wrong shape, cut at the token cap, a failed call)
gives no result, so the caller searches with the original question. `BudgetExceeded` propagates.
"""

import json
from collections.abc import Callable
from typing import Any, TypeVar

from production_rag.core.llm import BudgetExceeded, Usage

T = TypeVar("T")

MAX_QUERY_CHARS = 2000


async def ask(
    llm,
    messages: list[dict[str, str]],
    *,
    schema: dict,
    max_tokens: int,
    parse: Callable[[Any], T | None],
) -> tuple[T | None, Usage]:
    try:
        out = await llm.complete(messages, max_tokens=max_tokens, json_schema=schema)
    except BudgetExceeded:
        raise
    except Exception:  # a failed call falls back to the original question
        return None, Usage()
    if out.finish_reason == "length":
        return None, out.usage
    try:
        data = json.loads(out.text)
    except ValueError:
        return None, out.usage
    return parse(data), out.usage


def merge_queries(question: str, extra: list[str], limit: int = 8) -> list[str]:
    """The question first, then the extra queries without repeats, at most `limit` in all."""
    seen = {question.strip().casefold()}
    merged = [question]
    for q in extra:
        q = q.strip()[:MAX_QUERY_CHARS]
        if q and q.casefold() not in seen and len(merged) < limit:
            seen.add(q.casefold())
            merged.append(q)
    return merged

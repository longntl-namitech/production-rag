"""Ask an LLM for rephrasings of a question, for `multi_query.search`.

Any output that cannot be used (not JSON, wrong shape, cut at the token cap, a failed call)
gives no rephrasings, so the caller searches with the original question alone.
`BudgetExceeded` propagates.
"""

import json

from production_rag.core.llm import BudgetExceeded, Usage

MAX_TOKENS = 256
MAX_QUERY_CHARS = 2000

SCHEMA = {
    "name": "rewrites",
    "schema": {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
        "additionalProperties": False,
    },
}


def rewrite_messages(question: str, n: int = 3) -> list[dict[str, str]]:
    system = f"""You write search queries for a document search engine.
Given a question, write {n} different rephrasings that could find passages answering it.
- Keep the meaning of the question; do not answer it.
- Vary the wording: synonyms, a shorter keyword form, a more specific form.
- Write every query in the same language as the question.
Reply in JSON as {{"queries": [...]}}."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


async def rewrite(llm, question: str, n: int = 3) -> tuple[list[str], Usage]:
    try:
        out = await llm.complete(
            rewrite_messages(question, n), max_tokens=MAX_TOKENS, json_schema=SCHEMA
        )
    except BudgetExceeded:
        raise
    except Exception:  # a failed call falls back to the original question
        return [], Usage()
    if out.finish_reason == "length":
        return [], out.usage
    try:
        data = json.loads(out.text)
    except ValueError:
        return [], out.usage
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        return [], out.usage
    seen = {question.strip().casefold()}
    kept: list[str] = []
    for q in queries:
        if not isinstance(q, str) or not q.strip():
            continue
        q = q.strip()[:MAX_QUERY_CHARS]
        if q.casefold() not in seen:
            seen.add(q.casefold())
            kept.append(q)
    return kept[:n], out.usage

"""Split a statement into self-contained checkable claims, each with a search query."""

import json
from dataclasses import dataclass

from production_rag.core.llm import AuthFailed, BudgetExceeded, Usage, is_auth_error

MAX_TOKENS = 512
MAX_CLAIMS = 5

PROMPT = """You list the checkable factual claims in a statement.
- One fact per claim.
- Make each claim self-contained: spell out names and numbers, use no pronouns.
- Skip opinions, questions, greetings and promises.
- Give at most {max_claims} claims, each with a short search query that would find evidence.
- Write claims and queries in the language of the statement.
- When nothing is checkable, return an empty list.
Reply in JSON: {{"claims": [{{"claim": "...", "query": "..."}}]}}"""

SCHEMA = {
    "name": "claims",
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"claim": {"type": "string"}, "query": {"type": "string"}},
                    "required": ["claim", "query"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Claim:
    text: str
    query: str


@dataclass(frozen=True)
class Extraction:
    claims: list[Claim]
    usage: Usage
    error: str | None = None


def claim_messages(statement: str, max_claims: int = MAX_CLAIMS) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT.format(max_claims=max_claims)},
        {"role": "user", "content": statement},
    ]


async def extract_claims(llm, statement: str, max_claims: int = MAX_CLAIMS) -> Extraction:
    """`BudgetExceeded` propagates, a 401 or 403 raises `AuthFailed`, other failures set `error`."""
    messages = claim_messages(statement, max_claims)
    try:
        out = await llm.complete(messages, max_tokens=MAX_TOKENS, json_schema=SCHEMA)
    except BudgetExceeded:
        raise
    except Exception as exc:
        if is_auth_error(exc):
            raise AuthFailed(str(exc)) from None
        return Extraction([], Usage(), "llm_error")
    bad = Extraction([], out.usage, "bad_output")
    if out.finish_reason == "length":
        return bad
    try:
        data = json.loads(out.text)
    except ValueError:
        return bad
    items = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return bad
    claims: list[Claim] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            return bad
        text = item["claim"].strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        query = item.get("query")
        claims.append(
            Claim(text, query.strip() if isinstance(query, str) and query.strip() else text)
        )
    return Extraction(claims[:max_claims], out.usage)

"""Ask an LLM to split a question into sub-questions, for `query_decomposition.search`.

Any output that cannot be used, or an empty list, gives the question alone, so the caller still
searches. `BudgetExceeded` propagates.
"""

from collections.abc import Callable

from production_rag.core.llm import Usage
from production_rag.core.query_llm import MAX_QUERY_CHARS, ask

MAX_SUB = 3
MAX_TOKENS = 256

SCHEMA = {
    "name": "sub_questions",
    "schema": {
        "type": "object",
        "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
        "required": ["sub_questions"],
        "additionalProperties": False,
    },
}


def decompose_messages(question: str, n: int = MAX_SUB) -> list[dict[str, str]]:
    system = f"""You write search queries for a document search engine.
Split the question into at most {n} sub-questions, one for each independent part, so that each
can be answered by a different passage.
- If the question has one part, return it as the only sub-question.
- Each sub-question must make sense on its own; do not answer them.
- Write every sub-question in the same language as the question.
Reply in JSON as {{"sub_questions": [...]}}."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _parser(n: int) -> Callable[[object], list[str] | None]:
    def parse(data: object) -> list[str] | None:
        items = data.get("sub_questions") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return None
        seen: set[str] = set()
        kept: list[str] = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            text = item.strip()[:MAX_QUERY_CHARS]
            if text.casefold() not in seen:
                seen.add(text.casefold())
                kept.append(text)
        return kept[:n] or None

    return parse


async def queries_for(llm, question: str, n: int = MAX_SUB) -> tuple[list[str], Usage]:
    """The queries to search: the sub-questions (the question itself is not added)."""
    subs, usage = await ask(
        llm,
        decompose_messages(question, n),
        schema=SCHEMA,
        max_tokens=MAX_TOKENS,
        parse=_parser(n),
    )
    return (subs or [question]), usage

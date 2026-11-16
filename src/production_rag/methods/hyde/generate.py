"""Ask an LLM for a hypothetical passage that answers a question, for `hyde.search`.

Any output that cannot be used gives the original question, so the caller still searches.
`BudgetExceeded` propagates.
"""

from production_rag.core.llm import Usage
from production_rag.core.query_llm import ask

MAX_TOKENS = 256

SCHEMA = {
    "name": "hypothetical",
    "schema": {
        "type": "object",
        "properties": {"passage": {"type": "string"}},
        "required": ["passage"],
        "additionalProperties": False,
    },
}


def hyde_messages(question: str) -> list[dict[str, str]]:
    system = """You write a short passage for a document search engine.
Given a question, write the passage a reference document would contain to answer it.
- Two to four sentences, stated plainly, with no hedging and no mention of the question.
- Use the facts you know; a plausible passage is fine, the search finds the real one.
- Write in the same language as the question.
Reply in JSON as {"passage": "..."}."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _passage(data) -> str | None:
    passage = data.get("passage") if isinstance(data, dict) else None
    if isinstance(passage, str) and passage.strip():
        return passage.strip()
    return None


async def queries_for(
    llm, question: str, *, include_question: bool = False
) -> tuple[list[str], Usage]:
    """The queries to search: the passage, with the question first if asked; else the question."""
    passage, usage = await ask(
        llm, hyde_messages(question), schema=SCHEMA, max_tokens=MAX_TOKENS, parse=_passage
    )
    if passage is None:
        return [question], usage
    return ([question, passage] if include_question else [passage]), usage

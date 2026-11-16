"""Ask an LLM for one more general question, for `step_back.search`.

Any output that cannot be used, or a step-back that repeats the question, gives the question
alone, so the caller still searches. `BudgetExceeded` propagates.
"""

from production_rag.core.llm import Usage
from production_rag.core.query_llm import ask, merge_queries

MAX_TOKENS = 128

SCHEMA = {
    "name": "step_back",
    "schema": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": False,
    },
}


def step_back_messages(question: str) -> list[dict[str, str]]:
    system = """You write search queries for a document search engine.
Given a specific question, write one more general question about the topic or principle behind it,
the kind whose answer gives the background needed to answer the specific one.
- Do not answer the question and do not repeat it.
- Write in the same language as the question.
Reply in JSON as {"question": "..."}."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _question(data) -> str | None:
    text = data.get("question") if isinstance(data, dict) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


async def queries_for(llm, question: str) -> tuple[list[str], Usage]:
    """The queries to search: the question, then the step-back question if there is one."""
    step, usage = await ask(
        llm, step_back_messages(question), schema=SCHEMA, max_tokens=MAX_TOKENS, parse=_question
    )
    return merge_queries(question, [step] if step else []), usage

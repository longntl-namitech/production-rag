"""Ask an LLM whether the retrieved passages are enough to answer the question.

The label is the model's own judgement, not a probability. A reply that cannot be used (not
JSON, a label outside the two, a reply cut at the token cap, a failed call) has no label.
No passages is `insufficient` without a call: that is an answer, not a failure.
`BudgetExceeded` propagates.
"""

from production_rag.core.llm import Usage
from production_rag.core.models import Row
from production_rag.core.query_llm import ask

MAX_TOKENS = 64
LABELS = ("sufficient", "insufficient")

SCHEMA = {
    "name": "sufficiency",
    "schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": list(LABELS)}},
        "required": ["label"],
        "additionalProperties": False,
    },
}


def sufficiency_messages(question: str, rows: list[Row]) -> list[dict[str, str]]:
    system = """You check search results before a question is answered from them.
Say whether the passages contain enough to answer the question, judging only from the
passage text and not from what you know.
Reply in JSON as {"label": "sufficient"} or {"label": "insufficient"}."""
    passages = "\n\n".join(f"Passage {p}:\n{row.text}" for p, row in enumerate(rows, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\n{passages}"},
    ]


def _parse(data) -> str | None:
    label = data.get("label") if isinstance(data, dict) else None
    return label if label in LABELS else None


async def judge_sufficiency(llm, question: str, rows: list[Row]) -> tuple[str | None, Usage]:
    """`sufficient`, `insufficient`, or None when the model gave no usable label."""
    if not rows:
        return "insufficient", Usage()
    return await ask(
        llm,
        sufficiency_messages(question, rows),
        schema=SCHEMA,
        max_tokens=MAX_TOKENS,
        parse=_parse,
    )

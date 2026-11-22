"""Ask an LLM to score each retrieved passage from 0 to 10, then sort by that score.

The score is the model's own judgement, not a calibrated probability. A reply that cannot be
used (not JSON, a passage missing or repeated, an id or score out of range, a reply cut at the
token cap, a failed call) returns the passages in their base order. `BudgetExceeded` propagates.
"""

from production_rag.core.llm import Usage
from production_rag.core.models import Row
from production_rag.core.query_llm import ask

CANDIDATES = 20
MAX_TOKENS = 512
MAX_SCORE = 10
PASSAGE_TOKENS = 250  # upper bound for one 800-character chunk
PROMPT_TOKENS = 300

SCHEMA = {
    "name": "scores",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "score": {"type": "integer"}},
                    "required": ["id", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}


def rerank_messages(question: str, rows: list[Row]) -> list[dict[str, str]]:
    system = f"""You rate search results for a question.
Each passage is numbered. Give every passage a score from 0 to {MAX_SCORE}:
0 means unrelated or only sharing keywords with the question, {MAX_SCORE} means it answers
the question. Judge only from the passage text.
Reply in JSON as {{"scores": [{{"id": 1, "score": 7}}, {{"id": 2, "score": 0}}]}}
with every passage exactly once."""
    passages = "\n\n".join(f"Passage {p}:\n{row.text}" for p, row in enumerate(rows, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\n{passages}"},
    ]


def _parser(count: int):
    def parse(data) -> dict[int, int] | None:
        items = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != count:
            return None
        scores: dict[int, int] = {}
        for item in items:
            if not isinstance(item, dict):
                return None
            pid, score = item.get("id"), item.get("score")
            # `type is int` also rejects JSON booleans, which are ints in Python
            if type(pid) is not int or type(score) is not int:
                return None
            if not 1 <= pid <= count or pid in scores or not 0 <= score <= MAX_SCORE:
                return None
            scores[pid] = score
        return scores

    return parse


async def rerank(llm, question: str, rows: list[Row]) -> tuple[list[Row], Usage]:
    """The rows sorted by score, ties in base order; the rows unchanged when nothing is usable."""
    if not rows:
        return rows, Usage()
    scores, usage = await ask(
        llm,
        rerank_messages(question, rows),
        schema=SCHEMA,
        max_tokens=MAX_TOKENS,
        parse=_parser(len(rows)),
    )
    if scores is None:
        return rows, usage
    order = sorted(range(1, len(rows) + 1), key=lambda pid: -scores[pid])  # stable: ties keep order
    return [rows[pid - 1] for pid in order], usage


def estimate_tokens_per_query() -> int:
    """Upper bound of LLM tokens for one query."""
    return CANDIDATES * PASSAGE_TOKENS + PROMPT_TOKENS + MAX_TOKENS

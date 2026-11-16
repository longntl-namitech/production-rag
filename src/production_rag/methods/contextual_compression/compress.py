"""Ask an LLM which sentences of the retrieved passages help answer the question.

One call numbers every sentence of every passage (`1.1`, `1.2`, `2.1`, ...) and the LLM replies
with the ids to keep, so the text that comes back is always the original sentences, never new
words. A reply that cannot be used, a failed call, an empty list or no valid id returns the
passages whole. `BudgetExceeded` propagates.
"""

import re

from production_rag.core.llm import Usage
from production_rag.core.models import Row
from production_rag.core.query_llm import ask
from production_rag.core.semantic_chunking import split_sentences

MAX_TOKENS = 256

SCHEMA = {
    "name": "keep",
    "schema": {
        "type": "object",
        "properties": {"keep": {"type": "array", "items": {"type": "string"}}},
        "required": ["keep"],
        "additionalProperties": False,
    },
}

_ID = re.compile(r"^(\d+)\.(\d+)$")


def compress_messages(question: str, rows: list[Row]) -> list[dict[str, str]]:
    system = """You trim search results before they are used to answer a question.
Each passage is split into numbered sentences, written [passage.sentence].
Reply with the ids of the sentences that help answer the question, so that nothing needed for
the answer is lost and sentences that do not help are dropped.
Reply in JSON as {"keep": ["1.2", "2.1"]}."""
    lines: list[str] = []
    for p, row in enumerate(rows, 1):
        lines.append(f"Passage {p}:")
        lines += [f"[{p}.{s}] {text}" for s, text in enumerate(split_sentences(row.text), 1)]
    user = f"Question: {question}\n\n" + "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parser(sizes: list[int]):
    def parse(data) -> set[tuple[int, int]] | None:
        items = data.get("keep") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return None
        keep: set[tuple[int, int]] = set()
        for item in items:
            match = _ID.match(item) if isinstance(item, str) else None
            if match:
                p, s = int(match.group(1)), int(match.group(2))
                if 1 <= p <= len(sizes) and 1 <= s <= sizes[p - 1]:
                    keep.add((p, s))
        return keep or None

    return parse


async def compress(llm, question: str, rows: list[Row]) -> tuple[list[Row], Usage]:
    """The rows cut down to the kept sentences; the rows as they were when nothing is usable."""
    sentences = [split_sentences(row.text) for row in rows]
    keep, usage = await ask(
        llm,
        compress_messages(question, rows),
        schema=SCHEMA,
        max_tokens=MAX_TOKENS,
        parse=_parser([len(s) for s in sentences]),
    )
    if not keep:
        return rows, usage
    out: list[Row] = []
    for p, (row, parts) in enumerate(zip(rows, sentences, strict=True), 1):
        kept = [text for s, text in enumerate(parts, 1) if (p, s) in keep]
        if kept:
            out.append(row.model_copy(update={"text": " ".join(kept)}))
    return out, usage

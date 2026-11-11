import time
from dataclasses import dataclass

from production_rag.core.llm import Usage
from production_rag.core.models import Row

NO_ANSWER = "NO_ANSWER"
_WRAPPING = " \t\r\n[]()\"'`"

SYSTEM_PROMPT = f"""You answer a question using only the numbered passages provided.
- Use nothing but the passages. Do not rely on outside knowledge.
- Answer in the language of the question.
- Be short and direct. If the question is a yes/no question, start with "yes" or "no".
- If the passages do not contain the answer, reply with exactly {NO_ANSWER} and nothing else.
- Do not cite passage numbers and do not add any preamble."""


def is_abstention(text: str) -> bool:
    """True when the reply is NO_ANSWER, allowing wrapping brackets, quotes or a final period."""
    return text.strip(_WRAPPING).removesuffix(".").strip(_WRAPPING) == NO_ANSWER


@dataclass(frozen=True)
class Generated:
    text: str
    abstained: bool
    usage: Usage
    ms: int
    finish_reason: str


def build_messages(question: str, rows: list[Row]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[{i}] {r.title}\n{r.text}" if r.title else f"[{i}] {r.text}"
        for i, r in enumerate(rows, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{passages}\n\nQuestion: {question}"},
    ]


async def generate(llm, question: str, rows: list[Row], *, max_tokens: int = 512) -> Generated:
    if not rows:
        return Generated(NO_ANSWER, True, Usage(), 0, "no_passages")
    start = time.perf_counter()
    out = await llm.complete(build_messages(question, rows), max_tokens=max_tokens)
    ms = int((time.perf_counter() - start) * 1000)
    return Generated(out.text, is_abstention(out.text), out.usage, ms, out.finish_reason)

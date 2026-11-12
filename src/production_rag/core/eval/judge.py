"""LLM judges for the answer eval: correctness against a gold answer, faithfulness to passages.

A verdict the judge cannot give cleanly (invalid JSON, an unknown label, a reply cut at the
token cap, a failed call) has `label=None` and an `error`, so it is never scored as wrong.
`BudgetExceeded` propagates: the run stops instead of recording a judge failure. A 401 or 403
raises `AuthFailed`, since every further call would fail the same way.
"""

import json
from dataclasses import dataclass

from production_rag.core.llm import AuthFailed, BudgetExceeded, Usage, is_auth_error
from production_rag.core.models import Row

SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
CORRECTNESS_LABELS = tuple(SCORES)
FAITHFULNESS_LABELS = ("faithful", "unfaithful")
MAX_TOKENS = 256

CORRECTNESS_PROMPT = """You grade an answer to a question against a reference answer.
Compare meaning only. Ignore wording, style, length, language and extra details that are
correct and do not contradict the reference.
Labels:
- correct: the answer states what the reference states.
- partially_correct: the answer gets part of the reference right but misses or gets wrong
  another part of it.
- incorrect: the answer contradicts the reference, misses its point, or declines to answer.
Reply in JSON with a label and a one-sentence reason."""

FAITHFULNESS_PROMPT = """You check whether an answer to a question is supported by the numbered
passages. Read the answer as a reply to the question: a short answer such as "yes", "no", a name
or a number makes the claim the question asks about.
Labels:
- faithful: every claim the answer makes, read as a reply to the question, is stated in or
  follows directly from the passages.
- unfaithful: at least one such claim is missing from the passages or contradicts them.
Judge support only, not whether the answer is true in the world or complete.
Reply in JSON with a label and a one-sentence reason."""


@dataclass(frozen=True)
class Verdict:
    label: str | None
    reason: str
    usage: Usage
    error: str | None = None


def _schema(name: str, labels: tuple[str, ...]) -> dict:
    return {
        "name": name,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(labels)},
                "reason": {"type": "string"},
            },
            "required": ["label", "reason"],
            "additionalProperties": False,
        },
    }


def format_passages(rows: list[Row]) -> str:
    return "\n\n".join(
        f"[{i}] {r.title}\n{r.text}" if r.title else f"[{i}] {r.text}"
        for i, r in enumerate(rows, start=1)
    )


def correctness_messages(question: str, gold: str, answer: str) -> list[dict[str, str]]:
    user = f"Question: {question}\n\nReference answer: {gold}\n\nAnswer to grade: {answer}"
    return [
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {"role": "user", "content": user},
    ]


def faithfulness_messages(question: str, answer: str, rows: list[Row]) -> list[dict[str, str]]:
    passages = format_passages(rows)
    user = f"Question: {question}\n\nPassages:\n\n{passages}\n\nAnswer to check: {answer}"
    return [
        {"role": "system", "content": FAITHFULNESS_PROMPT},
        {"role": "user", "content": user},
    ]


async def judge_labels(
    llm,
    messages: list[dict[str, str]],
    name: str,
    labels: tuple[str, ...],
    max_tokens: int = MAX_TOKENS,
) -> Verdict:
    """One label + reason call. Shared by the answer judges and the fact-check verdict."""
    try:
        out = await llm.complete(messages, max_tokens=max_tokens, json_schema=_schema(name, labels))
    except BudgetExceeded:
        raise
    except Exception as exc:
        if is_auth_error(exc):
            raise AuthFailed(str(exc)) from None
        # any failed call (HTTP error, malformed body) is a judge failure
        return Verdict(None, "", Usage(), "llm_error")
    bad = Verdict(None, "", out.usage, "bad_output")
    if out.finish_reason == "length":
        return bad
    try:
        data = json.loads(out.text)
    except ValueError:
        return bad
    if not isinstance(data, dict) or data.get("label") not in labels:
        return bad
    reason = data.get("reason")
    return Verdict(data["label"], reason if isinstance(reason, str) else "", out.usage)


async def judge_correctness(llm, question: str, gold: str, answer: str) -> Verdict:
    messages = correctness_messages(question, gold, answer)
    return await judge_labels(llm, messages, "correctness", CORRECTNESS_LABELS)


async def judge_faithfulness(llm, question: str, answer: str, rows: list[Row]) -> Verdict:
    messages = faithfulness_messages(question, answer, rows)
    return await judge_labels(llm, messages, "faithfulness", FAITHFULNESS_LABELS)

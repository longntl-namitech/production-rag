"""Judge one claim against retrieved passages: supported, refuted or unverified."""

from production_rag.core.eval.judge import MAX_TOKENS, Verdict, format_passages, judge_labels
from production_rag.core.llm import Usage
from production_rag.core.models import Row

LABELS = ("supported", "refuted", "unverified")

PROMPT = """You decide whether numbered passages settle a claim. Use the passages only, not what
you know.
Labels:
- supported: the passages state the claim or directly imply it.
- refuted: the passages state something incompatible with the claim, including the opposite
  direction of an effect.
- unverified: the passages do not settle the claim or are about something else.
Judge the claim as written, not a nearby claim.
Reply in JSON with a label and a one-sentence reason."""

ClaimVerdict = Verdict  # label is None when the verdict failed; `error` says why


def verdict_messages(claim: str, rows: list[Row]) -> list[dict[str, str]]:
    user = f"Claim: {claim}\n\nPassages:\n\n{format_passages(rows)}"
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": user}]


async def judge_claim(llm, claim: str, rows: list[Row]) -> ClaimVerdict:
    """No rows -> unverified without a call. `BudgetExceeded` propagates, a 401 or 403 raises
    `AuthFailed`, other failures have `label=None` and an `error`."""
    if not rows:
        return ClaimVerdict("unverified", "no evidence retrieved", Usage())
    return await judge_labels(llm, verdict_messages(claim, rows), "verdict", LABELS, MAX_TOKENS)

"""Check the question for hazards first; retrieve only when none is flagged."""

from production_rag.core.deps import Deps
from production_rag.core.llm import Usage
from production_rag.core.models import SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods.guardrails.check import Verdict, check_input
from production_rag.methods.naive_rag import pipeline as naive_rag

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None


async def search(
    question: str,
    collection: str,
    deps: Deps,
    llm,
    *,
    topic: str,
    k: int = 5,
    deadline_ms: int = 4000,
) -> tuple[SpecResult, list[Verdict], Usage]:
    """No passages when a hazard is flagged; a failed result when a check gave no verdict.

    A check with no usable answer is not read as safe: nothing is retrieved (fail closed).
    """
    question = question[:2000]
    verdicts, usage = await check_input(llm, question, topic)
    if any(v.flagged for v in verdicts):
        return SpecResult(status="ok", elapsed_ms=0), verdicts, usage
    if any(v.flagged is None for v in verdicts):
        failed = SpecResult(status="failed", elapsed_ms=0, error="guardrail check failed")
        return failed, verdicts, usage
    result = await naive_rag.search(question, collection, deps, k=k, deadline_ms=deadline_ms)
    return result, verdicts, usage

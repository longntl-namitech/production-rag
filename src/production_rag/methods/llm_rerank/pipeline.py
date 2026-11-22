"""Retrieve 20 candidates, have an LLM score them, and keep the best k."""

from production_rag.core.deps import Deps
from production_rag.core.llm import Usage
from production_rag.core.models import SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods.llm_rerank.rerank import CANDIDATES, rerank
from production_rag.methods.naive_rag import pipeline as naive_rag

__all__ = ["CANDIDATES", "DEFAULT", "VARIANTS", "search"]

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None


async def search(
    question: str,
    collection: str,
    deps: Deps,
    llm,
    *,
    k: int = 5,
    deadline_ms: int = 4000,
) -> tuple[SpecResult, Usage]:
    """Candidates come from `naive_rag.search`; the LLM call afterwards only reorders them."""
    result = await naive_rag.search(
        question[:2000], collection, deps, k=CANDIDATES, deadline_ms=deadline_ms
    )
    if result.status != "ok" or not result.rows:
        return result, Usage()
    rows, usage = await rerank(llm, question[:2000], result.rows)
    return result.model_copy(update={"rows": rows[:k]}), usage

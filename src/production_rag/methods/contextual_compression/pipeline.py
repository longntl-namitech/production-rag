"""Retrieve the top passages, then keep only the sentences that help answer the question."""

from production_rag.core.deps import Deps
from production_rag.core.llm import Usage
from production_rag.core.models import SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods.contextual_compression.compress import compress
from production_rag.methods.naive_rag import pipeline as naive_rag

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
    """Retrieval is `naive_rag.search`; the LLM call afterwards only removes text."""
    result = await naive_rag.search(question, collection, deps, k=k, deadline_ms=deadline_ms)
    if result.status != "ok" or not result.rows:
        return result, Usage()
    rows, usage = await compress(llm, question, result.rows)
    return result.model_copy(update={"rows": rows}), usage

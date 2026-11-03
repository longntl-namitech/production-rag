"""Add neighbouring chunks (or the whole document) around each hit."""

from production_rag.core.deps import Deps
from production_rag.core.models import ContextOption, SearchRequest, SearchSpec, SpecResult
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search
from production_rag.core.registry import MethodConfig
from production_rag.methods import get_variant

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None


async def search(
    query: str,
    collection: str,
    deps: Deps,
    *,
    context: ContextOption = ContextOption(prev=1, next=1),  # noqa: B008
    base: str = "hybrid_weighted_rrf",
    k: int = 10,
    deadline_ms: int = 4000,
) -> SpecResult:
    req = SearchRequest(
        query=query,
        collection=collection,
        specs=[SearchSpec(name="default", k=k, context=context)],
    )
    response = await run_search(req, get_variant(base), deps, Deadline(deadline_ms))
    return response.results["default"]

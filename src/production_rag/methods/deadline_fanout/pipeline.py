"""Independent search specs under one shared deadline."""

from production_rag.core.deps import Deps
from production_rag.core.models import SearchRequest, SearchResponse, SearchSpec
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
    specs: list[SearchSpec],
    *,
    base: str = "hybrid_weighted_rrf",
    deadline_ms: int = 4000,
) -> SearchResponse:
    req = SearchRequest(query=query, collection=collection, specs=specs)
    return await run_search(req, get_variant(base), deps, Deadline(deadline_ms))

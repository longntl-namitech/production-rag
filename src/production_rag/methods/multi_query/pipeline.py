"""Several rephrasings of one query, merged by best score."""

from production_rag.core.deps import Deps
from production_rag.core.models import SearchRequest, SearchSpec, SpecResult
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search
from production_rag.core.registry import MethodConfig
from production_rag.methods import get_variant

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None


async def search(
    queries: list[str],
    collection: str,
    deps: Deps,
    *,
    rerank_query: str | None = None,
    base: str = "hybrid_weighted_rrf",
    k: int = 10,
    deadline_ms: int = 4000,
) -> SpecResult:
    req = SearchRequest(
        query=queries,
        rerank_query=rerank_query,
        collection=collection,
        specs=[SearchSpec(name="default", k=k)],
    )
    response = await run_search(req, get_variant(base), deps, Deadline(deadline_ms))
    return response.results["default"]

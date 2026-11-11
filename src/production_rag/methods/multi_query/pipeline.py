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
    variant = get_variant(base)
    if rerank_query is not None and not variant.rerank:
        raise ValueError("rerank_query needs a base that reranks, e.g. base='hybrid_rrf_rerank'")
    req = SearchRequest(
        query=queries,
        rerank_query=rerank_query,
        collection=collection,
        specs=[SearchSpec(name="default", k=k)],
    )
    response = await run_search(req, variant, deps, Deadline(deadline_ms))
    return response.results["default"]

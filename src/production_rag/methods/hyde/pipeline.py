"""Search with a hypothetical answer passage, on the dense vectors only."""

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
    base: str = "hybrid_weighted_rrf",
    k: int = 10,
    deadline_ms: int = 4000,
) -> SpecResult:
    """Search `queries` (from `generate.queries_for`) with dense vectors only.

    A hypothetical passage matches by meaning, not by words, so the sparse leg is skipped
    for every query in the list.
    """
    req = SearchRequest(
        query=queries,
        collection=collection,
        specs=[SearchSpec(name="default", k=k, search_type="dense")],
    )
    response = await run_search(req, get_variant(base), deps, Deadline(deadline_ms))
    return response.results["default"]

"""Shared body of the single-spec `search` functions in the variant folders."""

from production_rag.core.deps import Deps
from production_rag.core.models import SearchRequest, SearchSpec, SpecResult
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search
from production_rag.core.registry import MethodConfig


async def run_variant(
    variants: dict[str, MethodConfig],
    variant: str,
    query: str,
    collection: str,
    deps: Deps,
    k: int,
    deadline_ms: int,
) -> SpecResult:
    try:
        method = variants[variant]
    except KeyError:
        raise KeyError(f"unknown variant {variant!r}, known: {', '.join(variants)}") from None
    req = SearchRequest(query=query, collection=collection, specs=[SearchSpec(name="default", k=k)])
    response = await run_search(req, method, deps, Deadline(deadline_ms))
    return response.results["default"]

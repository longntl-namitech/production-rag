"""Dense vector search."""

from production_rag.core.deps import Deps
from production_rag.core.models import SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods._variant import run_variant

VARIANTS: dict[str, MethodConfig] = {
    "dense": MethodConfig("dense", ("dense",), "none", doc="dense vectors only"),
}

DEFAULT = "dense"


async def search(
    query: str,
    collection: str,
    deps: Deps,
    *,
    variant: str = DEFAULT,
    k: int = 10,
    deadline_ms: int = 4000,
) -> SpecResult:
    return await run_variant(VARIANTS, variant, query, collection, deps, k, deadline_ms)

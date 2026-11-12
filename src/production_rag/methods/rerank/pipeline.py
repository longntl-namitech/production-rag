"""Hybrid search followed by a rerank stage."""

from production_rag.core.deps import Deps
from production_rag.core.models import SpecResult
from production_rag.core.registry import Leg, MethodConfig
from production_rag.methods._variant import run_variant

_HYBRID: tuple[Leg, ...] = ("dense", "sparse")

VARIANTS: dict[str, MethodConfig] = {
    "hybrid_rrf_rerank": MethodConfig(
        "hybrid_rrf_rerank", _HYBRID, "rrf", rerank=True, doc="hybrid RRF, then rerank"
    ),
}

DEFAULT = "hybrid_rrf_rerank"


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

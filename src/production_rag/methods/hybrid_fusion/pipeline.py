"""Hybrid dense plus BM25 search with fusion."""

from production_rag.core.deps import Deps
from production_rag.core.models import SpecResult
from production_rag.core.registry import Leg, MethodConfig
from production_rag.methods._variant import run_variant

_HYBRID: tuple[Leg, ...] = ("dense", "sparse")

VARIANTS: dict[str, MethodConfig] = {
    "hybrid_rrf": MethodConfig(
        "hybrid_rrf", _HYBRID, "rrf", doc="dense + BM25, reciprocal rank fusion"
    ),
    "hybrid_dbsf": MethodConfig(
        "hybrid_dbsf", _HYBRID, "dbsf", doc="dense + BM25, distribution-based fusion"
    ),
    "hybrid_weighted_rrf": MethodConfig(
        "hybrid_weighted_rrf",
        _HYBRID,
        "weighted_rrf",
        weights=(1.0, 0.5),
        doc="dense + BM25, RRF with dense weighted higher",
    ),
}

DEFAULT = "hybrid_weighted_rrf"


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

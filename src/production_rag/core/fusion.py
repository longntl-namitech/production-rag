"""Turn a method's legs and fusion into Qdrant query_points arguments.

All fusion runs inside Qdrant: each leg is a prefetch, and the outer query
fuses them (RRF, DBSF, or RRF with per-leg weights).
"""

from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from production_rag.core.registry import Fusion, Leg


@dataclass
class QueryVectors:
    dense: list[float] | None
    sparse: models.SparseVector | None
    sparse_vi: models.SparseVector | None


def _vector_for(leg: Leg, vectors: QueryVectors) -> Any:
    vector = {
        "dense": vectors.dense,
        "sparse": vectors.sparse,
        "sparse_noidf": vectors.sparse,
        "sparse_vi": vectors.sparse_vi,
    }[leg]
    if vector is None:
        raise ValueError(f"no query vector for leg {leg!r}")
    return vector


def _fusion_query(fusion: Fusion, weights: tuple[float, ...] | None) -> Any:
    if fusion == "rrf":
        return models.FusionQuery(fusion=models.Fusion.RRF)
    if fusion == "dbsf":
        return models.FusionQuery(fusion=models.Fusion.DBSF)
    if fusion == "weighted_rrf":
        return models.RrfQuery(rrf=models.Rrf(weights=list(weights or ())))
    raise ValueError(f"fusion {fusion!r} needs more than one leg")


def build_query(
    legs: tuple[Leg, ...],
    fusion: Fusion,
    weights: tuple[float, ...] | None,
    vectors: QueryVectors,
    prefetch_limit: int,
    flt: models.Filter | None,
) -> dict[str, Any]:
    if len(legs) == 1:
        return {"query": _vector_for(legs[0], vectors), "using": legs[0], "query_filter": flt}
    prefetch = [
        models.Prefetch(
            query=_vector_for(leg, vectors), using=leg, limit=prefetch_limit, filter=flt
        )
        for leg in legs
    ]
    return {"prefetch": prefetch, "query": _fusion_query(fusion, weights)}

"""Named retrieval methods: which vector legs to search and how to fuse them.

Adding a method means adding a MethodConfig here; the pipeline and the eval
harness pick it up by name.
"""

from dataclasses import dataclass
from typing import Literal

Leg = Literal["dense", "sparse", "sparse_noidf", "sparse_vi"]
Fusion = Literal["none", "rrf", "dbsf", "weighted_rrf"]

SPARSE_LEGS: tuple[Leg, ...] = ("sparse", "sparse_noidf", "sparse_vi")


@dataclass(frozen=True)
class MethodConfig:
    name: str
    legs: tuple[Leg, ...]
    fusion: Fusion
    weights: tuple[float, ...] | None = None
    rerank: bool = False
    doc: str = ""


_HYBRID: tuple[Leg, ...] = ("dense", "sparse")

METHODS: dict[str, MethodConfig] = {
    m.name: m
    for m in [
        MethodConfig("dense", ("dense",), "none", doc="dense vectors only"),
        MethodConfig("bm25_no_idf", ("sparse_noidf",), "none", doc="BM25 without IDF"),
        MethodConfig("bm25", ("sparse",), "none", doc="BM25 with IDF"),
        MethodConfig(
            "bm25_vi_stopwords", ("sparse_vi",), "none", doc="BM25, Vietnamese stopwords removed"
        ),
        MethodConfig("hybrid_rrf", _HYBRID, "rrf", doc="dense + BM25, reciprocal rank fusion"),
        MethodConfig("hybrid_dbsf", _HYBRID, "dbsf", doc="dense + BM25, distribution-based fusion"),
        MethodConfig(
            "hybrid_weighted_rrf",
            _HYBRID,
            "weighted_rrf",
            weights=(1.0, 0.5),
            doc="dense + BM25, RRF with dense weighted higher",
        ),
        MethodConfig(
            "hybrid_rrf_rerank", _HYBRID, "rrf", rerank=True, doc="hybrid RRF, then rerank"
        ),
    ]
}


def get_method(name: str) -> MethodConfig:
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}, known: {', '.join(METHODS)}") from None


def legs_for(method: MethodConfig, search_type: str) -> tuple[Leg, ...]:
    if search_type == "dense":
        return ("dense",)
    if search_type == "sparse":
        sparse = tuple(leg for leg in method.legs if leg in SPARSE_LEGS)
        return sparse or ("sparse",)
    return method.legs

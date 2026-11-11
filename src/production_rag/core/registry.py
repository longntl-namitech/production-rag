"""Method configuration: which vector legs to search and how to fuse them.

The named variants live in the method folders under production_rag.methods.
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
    chunk_chars: int | None = None


def legs_for(method: MethodConfig, search_type: str) -> tuple[Leg, ...]:
    if search_type == "dense":
        return ("dense",)
    if search_type == "sparse":
        sparse = tuple(leg for leg in method.legs if leg in SPARSE_LEGS)
        return sparse or ("sparse",)
    return method.legs

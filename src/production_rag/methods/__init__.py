"""One folder per method. Variant names are the metric names used by the eval."""

import importlib

from production_rag.core.registry import MethodConfig

METHOD_NAMES: tuple[str, ...] = (
    "dense",
    "bm25",
    "hybrid_fusion",
    "rerank",
    "multi_query",
    "context_expansion",
    "deadline_fanout",
)


def _folder_variants(name: str) -> dict[str, MethodConfig]:
    module = importlib.import_module(f"production_rag.methods.{name}.pipeline")
    return module.VARIANTS


def all_variants() -> dict[str, MethodConfig]:
    merged: dict[str, MethodConfig] = {}
    for name in METHOD_NAMES:
        merged.update(_folder_variants(name))
    return merged


def get_variant(name: str) -> MethodConfig:
    variants = all_variants()
    try:
        return variants[name]
    except KeyError:
        raise KeyError(f"unknown variant {name!r}, known: {', '.join(variants)}") from None


def resolve(names: list[str]) -> list[str]:
    """Expand folder names to their variants; variant names pass through."""
    variants = all_variants()
    out: list[str] = []
    for name in names:
        if name in METHOD_NAMES:
            expanded = list(_folder_variants(name))
            if not expanded:
                raise ValueError(f"{name!r} has no eval variants")
        elif name in variants:
            expanded = [name]
        else:
            known = ", ".join([*METHOD_NAMES, *variants])
            raise ValueError(f"unknown method {name!r}, known: {known}")
        out.extend(v for v in expanded if v not in out)
    return out

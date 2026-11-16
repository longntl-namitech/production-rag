"""Search the question and its step-back question: the same merge as `multi_query`."""

from production_rag.core.registry import MethodConfig
from production_rag.methods.multi_query.pipeline import search

__all__ = ["DEFAULT", "VARIANTS", "search"]

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None

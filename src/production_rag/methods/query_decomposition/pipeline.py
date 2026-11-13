"""Search the sub-questions: the same merge as `multi_query`.

When the base reranks, pass `rerank_query=<the original question>` so the reranker judges
against the whole question and not one sub-question.
"""

from production_rag.core.registry import MethodConfig
from production_rag.methods.multi_query.pipeline import search

__all__ = ["DEFAULT", "VARIANTS", "search"]

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None

"""Retrieve the top passages, then ask an LLM whether they are enough to answer the question."""

from dataclasses import dataclass

from production_rag.core.deps import Deps
from production_rag.core.llm import Usage
from production_rag.core.models import Row
from production_rag.core.registry import MethodConfig
from production_rag.methods import all_variants
from production_rag.methods._variant import run_variant
from production_rag.methods.retrieval_sufficiency.judge import judge_sufficiency

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None

BASE = "hybrid_weighted_rrf"


@dataclass(frozen=True)
class Sufficiency:
    label: str | None  # "sufficient", "insufficient", or None when there is no usable verdict
    rows: list[Row]
    usage: Usage
    error: str | None = None


async def search(
    question: str,
    collection: str,
    deps: Deps,
    llm,
    *,
    base: str = BASE,
    k: int = 5,
    deadline_ms: int = 4000,
) -> Sufficiency:
    """The passages of `base` and the model's verdict on them; nothing is searched again."""
    question = question[:2000]
    result = await run_variant(all_variants(), base, question, collection, deps, k, deadline_ms)
    if result.status != "ok":
        return Sufficiency(None, [], Usage(), error=result.error or "retrieval failed")
    label, usage = await judge_sufficiency(llm, question, result.rows)
    return Sufficiency(label, result.rows, usage, error=None if label else "unusable reply")

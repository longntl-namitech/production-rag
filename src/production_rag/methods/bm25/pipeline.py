"""BM25 sparse search."""

from production_rag.core.deps import Deps
from production_rag.core.models import SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods._variant import run_variant

VARIANTS: dict[str, MethodConfig] = {
    "bm25_no_idf": MethodConfig("bm25_no_idf", ("sparse_noidf",), "none", doc="BM25 without IDF"),
    "bm25": MethodConfig("bm25", ("sparse",), "none", doc="BM25 with IDF"),
    "bm25_vi_stopwords": MethodConfig(
        "bm25_vi_stopwords", ("sparse_vi",), "none", doc="BM25, Vietnamese stopwords removed"
    ),
}

DEFAULT = "bm25"


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

"""Index chunks with an LLM-written context sentence, then search them like any other chunks."""

from production_rag.core.cache import DiskCache
from production_rag.core.chunking import chunk_documents
from production_rag.core.deps import Deps
from production_rag.core.indexing import index_chunks
from production_rag.core.llm import Usage
from production_rag.core.models import SearchRequest, SearchSpec, SpecResult
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search
from production_rag.core.registry import MethodConfig
from production_rag.methods import get_variant
from production_rag.methods.contextual_headers.context import add_context

VARIANTS: dict[str, MethodConfig] = {}

DEFAULT: str | None = None


async def index(
    deps: Deps,
    llm,
    cache: DiskCache,
    collection: str,
    docs: dict[str, tuple[str | None, str]],
    *,
    size: int = 800,
) -> tuple[int, Usage]:
    """Chunk the documents (recursive, `size` characters), add context, embed and store them."""
    chunks = chunk_documents(docs, size, strategy="recursive")
    with_context, usage = await add_context(llm, cache, docs, chunks)
    return await index_chunks(deps, collection, with_context), usage


async def search(
    query: str,
    collection: str,
    deps: Deps,
    *,
    variant: str = "chunk_recursive_800",
    k: int = 10,
    deadline_ms: int = 4000,
) -> SpecResult:
    req = SearchRequest(query=query, collection=collection, specs=[SearchSpec(name="default", k=k)])
    response = await run_search(req, get_variant(variant), deps, Deadline(deadline_ms))
    return response.results["default"]

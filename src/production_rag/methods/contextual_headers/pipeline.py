"""Index chunks with an LLM-written context sentence, then search them like any other chunks."""

from dataclasses import dataclass

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


class ContextIncomplete(RuntimeError):
    """Some chunks got no context sentence, so the collection was not written."""


@dataclass(frozen=True)
class IndexResult:
    count: int
    usage: Usage
    without_context: int


async def index(
    deps: Deps,
    llm,
    cache: DiskCache,
    collection: str,
    docs: dict[str, tuple[str | None, str]],
    *,
    size: int = 800,
) -> IndexResult:
    """Chunk the documents (recursive, `size` characters), add context, embed and store them.

    A collection that already holds every chunk is reused as is. If any chunk gets no context
    (a failed call), nothing is written and `ContextIncomplete` is raised: `index_chunks` keeps
    a collection whose point count matches, so a half-contextual one could never be repaired.
    What was generated stays in the cache, so the rerun only pays for the missing chunks.
    """
    chunks = chunk_documents(docs, size, strategy="recursive")
    await deps.store.ensure_collection(collection, deps.embedder.dim)
    if await deps.store.count(collection) == len(chunks):
        return IndexResult(len(chunks), Usage(), 0)
    with_context, usage = await add_context(llm, cache, docs, chunks)
    missing = sum(chunk.context is None for chunk in with_context)
    if missing:
        raise ContextIncomplete(
            f"{missing} of {len(chunks)} chunks got no context sentence; collection "
            f"{collection!r} not written. Fix the LLM problem and rerun (generated "
            "sentences are cached)."
        )
    return IndexResult(await index_chunks(deps, collection, with_context), usage, 0)


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

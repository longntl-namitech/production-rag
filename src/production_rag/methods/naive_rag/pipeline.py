"""Naive RAG baseline: 800-character chunks, dense top-k, one generation call."""

from dataclasses import dataclass

from production_rag.core.chunking import chunk_documents
from production_rag.core.deps import Deps
from production_rag.core.generation import generate
from production_rag.core.indexing import index_chunks
from production_rag.core.llm import Usage
from production_rag.core.models import Row, SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods._variant import run_variant

CHUNK_CHARS = 800

VARIANTS: dict[str, MethodConfig] = {
    "naive_rag": MethodConfig(
        "naive_rag", ("dense",), "none", chunk_chars=CHUNK_CHARS, doc="800-char chunks, dense top-k"
    ),
}

DEFAULT = "naive_rag"


@dataclass(frozen=True)
class RagAnswer:
    text: str
    abstained: bool
    rows: list[Row]
    usage: Usage


async def search(
    query: str,
    collection: str,
    deps: Deps,
    *,
    variant: str = DEFAULT,
    k: int = 5,
    deadline_ms: int = 4000,
) -> SpecResult:
    return await run_variant(VARIANTS, variant, query, collection, deps, k, deadline_ms)


async def index(deps: Deps, collection: str, docs: dict[str, tuple[str | None, str]]) -> int:
    """Split the documents into chunks and index them. Returns the chunk count."""
    return await index_chunks(deps, collection, chunk_documents(docs, CHUNK_CHARS))


async def answer(question: str, collection: str, deps: Deps, llm, *, k: int = 5) -> RagAnswer:
    result = await search(question, collection, deps, k=k)
    generated = await generate(llm, question, result.rows)
    return RagAnswer(generated.text, generated.abstained, result.rows, generated.usage)

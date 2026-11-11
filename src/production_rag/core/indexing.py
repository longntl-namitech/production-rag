import asyncio
from dataclasses import dataclass

from production_rag.core.deps import Deps
from production_rag.core.store import ChunkPoint


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    title: str | None = None


def _document(title: str | None, text: str) -> str:
    return f"{title}\n{text}" if title else text


async def index_chunks(
    deps: Deps, collection: str, chunks: list[Chunk], batch_size: int = 32
) -> int:
    """Embed and store the chunks. A complete collection is left as is."""
    await deps.store.ensure_collection(collection, deps.embedder.dim)
    if await deps.store.count(collection) == len(chunks):
        return len(chunks)
    for n, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        texts = [_document(c.title, c.text) for c in batch]
        dense = await deps.embedder.embed_documents(texts)
        sparse = await asyncio.to_thread(deps.bm25.encode_documents, texts)
        sparse_vi = await asyncio.to_thread(deps.bm25_vi.encode_documents, texts)
        await deps.store.upsert(
            collection,
            [
                ChunkPoint(
                    c.doc_id,
                    c.chunk_index,
                    c.text,
                    dense[i],
                    sparse[i],
                    sparse_vi[i],
                    title=c.title,
                )
                for i, c in enumerate(batch)
            ],
        )
        if n % 20 == 0:
            print(f"indexed {start + len(batch)}/{len(chunks)}", flush=True)
    return await deps.store.count(collection)

import asyncio

from production_rag.core.models import ContextOption, Row
from production_rag.core.store import Store


async def _expand_one(store: Store, collection: str, row: Row, option: ContextOption) -> Row:
    if option.full:
        chunks = await store.fetch_chunks(collection, row.doc_id, None, None)
    else:
        start = max(0, row.chunk_index - option.prev)
        chunks = await store.fetch_chunks(
            collection, row.doc_id, start, row.chunk_index + option.next
        )
    if not chunks:
        return row
    return row.model_copy(update={"text": "\n".join(c.text for c in chunks)})


async def expand_context(
    store: Store, collection: str, rows: list[Row], option: ContextOption
) -> list[Row]:
    """Replace each hit's text with its neighbouring chunks, or its whole document."""
    return list(await asyncio.gather(*(_expand_one(store, collection, r, option) for r in rows)))

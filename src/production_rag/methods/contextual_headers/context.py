"""Ask an LLM for a sentence that places a chunk in its document, for `contextual_headers`.

The sentence is embedded and BM25-encoded with the chunk; the stored chunk text is not
changed. Results go through a `DiskCache`, so a rerun costs nothing. A failed or empty
generation leaves the chunk without context and is not cached. `BudgetExceeded` propagates.
"""

import asyncio
from dataclasses import replace

from production_rag.core.cache import DiskCache
from production_rag.core.indexing import Chunk
from production_rag.core.llm import BudgetExceeded, Usage

PROMPT_VERSION = "v1"
MAX_TOKENS = 100
MAX_DOC_CHARS = 12000  # a longer document is cut here before it goes into the prompt

_SYSTEM = """You help a search engine find passages by placing each one in its document.
Given a document and one passage from it, write one short sentence that says where the passage
sits in the document and what it is about, using names and terms from the document.
- Reply with the sentence only: no preamble, no quotes.
- Write in the same language as the document."""


def context_messages(document: str, chunk: str) -> list[dict[str, str]]:
    user = f"<document>\n{document[:MAX_DOC_CHARS]}\n</document>\n\n<passage>\n{chunk}\n</passage>"
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


async def add_context(
    llm,
    cache: DiskCache,
    docs: dict[str, tuple[str | None, str]],
    chunks: list[Chunk],
    *,
    concurrency: int = 8,
) -> tuple[list[Chunk], Usage]:
    """The chunks with `context` set where a sentence could be made, and the tokens used."""
    semaphore = asyncio.Semaphore(concurrency)
    usage = Usage()

    async def one(chunk: Chunk) -> Chunk:
        nonlocal usage
        messages = context_messages(docs[chunk.doc_id][1], chunk.text)
        key = DiskCache.key(llm.model, PROMPT_VERSION + _SYSTEM, messages[1]["content"])
        cached = cache.get(key)
        if cached is not None:
            return replace(chunk, context=cached)
        async with semaphore:
            try:
                out = await llm.complete(messages, max_tokens=MAX_TOKENS)
            except BudgetExceeded:
                raise
            except Exception:  # the chunk is indexed without context
                return chunk
        usage += out.usage
        text = out.text.strip()
        if out.finish_reason == "length" or not text:
            return chunk
        cache.put(key, text)
        return replace(chunk, context=text)

    tasks = [asyncio.ensure_future(one(chunk)) for chunk in chunks]
    try:
        return list(await asyncio.gather(*tasks)), usage
    except BaseException:
        for task in tasks:
            task.cancel()
        raise

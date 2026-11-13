import pytest
from fakes import FakeLLM
from test_indexing import RecordingEmbedder

from production_rag.core.cache import DiskCache
from production_rag.core.indexing import Chunk
from production_rag.core.llm import BudgetExceeded, Completion, LLMError, Usage
from production_rag.methods import METHOD_NAMES
from production_rag.methods.contextual_headers import pipeline as contextual_headers
from production_rag.methods.contextual_headers.context import (
    MAX_DOC_CHARS,
    add_context,
    context_messages,
)
from production_rag.methods.contextual_headers.pipeline import ContextIncomplete

DOCS = {"d1": ("Leave", "Annual leave policy. Staff get twelve days. Sick leave is paid.")}
CHUNKS = [
    Chunk("d1", 0, "Annual leave policy.", "Leave"),
    Chunk("d1", 1, "Staff get twelve days.", "Leave"),
    Chunk("d1", 2, "Sick leave is paid.", "Leave"),
]


class Boom:
    model = "boom"

    async def complete(self, messages, *, max_tokens, json_schema=None):
        raise LLMError("x")


class BudgetAfter:
    """Answers the first `ok` calls, then runs out of budget."""

    model = "fake"

    def __init__(self, ok: int):
        self.ok = ok
        self.calls = 0

    async def complete(self, messages, *, max_tokens, json_schema=None):
        self.calls += 1
        if self.calls > self.ok:
            raise BudgetExceeded("full")
        return Completion("a context", Usage(10, 5), "stop", "fake")


async def test_add_context_sets_context_and_strips(tmp_path):
    llm = FakeLLM(lambda m: "  From the leave policy.  \n")
    out, usage = await add_context(llm, DiskCache(tmp_path), DOCS, CHUNKS)
    assert [c.context for c in out] == ["From the leave policy."] * 3
    assert [(c.doc_id, c.chunk_index, c.text) for c in out] == [
        (c.doc_id, c.chunk_index, c.text) for c in CHUNKS
    ]
    assert usage == Usage(30, 15)


async def test_add_context_uses_cache_second_time(tmp_path):
    llm = FakeLLM(lambda m: "From the leave policy.")
    cache = DiskCache(tmp_path)
    first, _ = await add_context(llm, cache, DOCS, CHUNKS)
    second, usage = await add_context(llm, cache, DOCS, CHUNKS)
    assert len(llm.calls) == 3 and second == first and usage == Usage()


async def test_failed_generation_is_not_cached_and_chunk_has_no_context(tmp_path):
    out, usage = await add_context(Boom(), DiskCache(tmp_path), DOCS, CHUNKS)
    assert [c.context for c in out] == [None, None, None] and usage == Usage()
    assert list(tmp_path.rglob("*.json")) == []


async def test_empty_or_truncated_reply_gives_no_context(tmp_path):
    empty, _ = await add_context(FakeLLM(lambda m: "  "), DiskCache(tmp_path), DOCS, CHUNKS[:1])
    cut = FakeLLM(lambda m: "half a sent", finish_reason="length")
    truncated, _ = await add_context(cut, DiskCache(tmp_path), DOCS, CHUNKS[:1])
    assert empty[0].context is None and truncated[0].context is None
    assert list(tmp_path.rglob("*.json")) == []


async def test_budget_exceeded_propagates_and_earlier_results_stay_cached(tmp_path):
    cache = DiskCache(tmp_path)
    try:
        await add_context(BudgetAfter(1), cache, DOCS, CHUNKS, concurrency=1)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("BudgetExceeded did not propagate")
    again = FakeLLM(lambda m: "a context")
    out, _ = await add_context(again, cache, DOCS, CHUNKS, concurrency=1)
    assert len(again.calls) == 2 and all(c.context == "a context" for c in out)


def test_document_is_cut_in_prompt():
    messages = context_messages("Z" * 50000, "the chunk")
    assert messages[-1]["content"].count("Z") == MAX_DOC_CHARS
    assert "the chunk" in messages[-1]["content"]


async def test_index_stores_raw_text_and_embeds_context(deps, tmp_path):
    deps.embedder = RecordingEmbedder()
    llm = FakeLLM(lambda m: "From the leave policy.")
    result = await contextual_headers.index(deps, llm, DiskCache(tmp_path), "ctx", DOCS, size=800)
    body = DOCS["d1"][1]
    assert result.count == 1 and result.usage == Usage(10, 5) and result.without_context == 0
    assert deps.embedder.texts == [f"Leave\nFrom the leave policy.\n{body}"]
    [row] = await deps.store.fetch_chunks("ctx", "d1", None, None)
    assert row.text == body


async def test_index_refuses_a_collection_with_chunks_missing_context(deps, tmp_path):
    with pytest.raises(ContextIncomplete, match="1 of 1"):
        await contextual_headers.index(deps, Boom(), DiskCache(tmp_path), "ctx", DOCS)
    assert await deps.store.count("ctx") == 0


async def test_index_rerun_repairs_after_a_failed_run(deps, tmp_path):
    deps.embedder = RecordingEmbedder()
    cache = DiskCache(tmp_path)
    with pytest.raises(ContextIncomplete):
        await contextual_headers.index(deps, Boom(), cache, "ctx", DOCS)
    llm = FakeLLM(lambda m: "From the leave policy.")
    result = await contextual_headers.index(deps, llm, cache, "ctx", DOCS)
    assert result.count == 1 and result.without_context == 0
    assert "From the leave policy." in deps.embedder.texts[-1]


async def test_index_reuses_a_complete_collection_without_llm_calls(deps, tmp_path):
    llm = FakeLLM(lambda m: "From the leave policy.")
    await contextual_headers.index(deps, llm, DiskCache(tmp_path / "a"), "ctx", DOCS)
    again = FakeLLM(lambda m: "x")
    result = await contextual_headers.index(deps, again, DiskCache(tmp_path / "b"), "ctx", DOCS)
    assert again.calls == [] and result.count == 1 and result.usage == Usage()


async def test_search_finds_the_chunk(deps, tmp_path):
    llm = FakeLLM(lambda m: "From the leave policy.")
    await contextual_headers.index(deps, llm, DiskCache(tmp_path), "ctx", DOCS)
    result = await contextual_headers.search("annual leave", "ctx", deps, k=1)
    assert result.rows[0].doc_id == "d1"


def test_contextual_headers_registered():
    assert "contextual_headers" in METHOD_NAMES
    assert contextual_headers.VARIANTS == {} and contextual_headers.DEFAULT is None

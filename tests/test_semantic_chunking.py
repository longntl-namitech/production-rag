from production_rag.core.semantic_chunking import (
    MAX_CHARS,
    MIN_CHARS,
    semantic_chunk_documents,
    split_sentences,
)

CATS = (
    "The cat sat on the warm mat. A cat purred in the sun. "
    "My cat slept all afternoon. Her cat chased a red ball."
)
STOCKS = (
    "Stock prices rose on Monday. Stock bonds fell by a little. "
    "Stock news was calm. Stock desks stayed busy."
)


class TopicEmbedder:
    """cat sentences point one way, stock sentences another, everything else between."""

    dim = 2

    def __init__(self):
        self.batches: list[int] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        out = []
        for text in texts:
            low = text.lower()
            out.append([1.0, 0.0] if "cat" in low else [0.0, 1.0] if "stock" in low else [1.0, 1.0])
        return out


def test_split_sentences_basic():
    assert split_sentences("A b. C d? E f!") == ["A b.", "C d?", "E f!"]
    assert split_sentences("one line\ntwo line\n\nthree line") == [
        "one line",
        "two line",
        "three line",
    ]
    assert split_sentences("") == []
    assert split_sentences("  \n ") == []


def test_split_sentences_vietnamese_keeps_text():
    text = "Điều 1. Nội dung quy định. Điều 2. Khác."
    parts = split_sentences(text)
    assert len(parts) == 4 and " ".join(parts) == text


async def test_semantic_chunks_split_at_topic_change():
    cats, stocks = CATS, STOCKS
    assert len(cats) >= MIN_CHARS and len(stocks) >= MIN_CHARS
    chunks = await semantic_chunk_documents({"d": ("T", f"{cats} {stocks}")}, TopicEmbedder())
    assert [c.text for c in chunks] == [cats, stocks]


async def test_semantic_chunk_cap():
    text = " ".join(f"The cat number {i} sat down." for i in range(100))
    chunks = await semantic_chunk_documents({"d": (None, text)}, TopicEmbedder())
    assert len(chunks) > 1
    assert all(0 < len(c.text) <= MAX_CHARS for c in chunks)


async def test_semantic_single_sentence_and_empty():
    one = await semantic_chunk_documents({"d": (None, "Only one cat sentence.")}, TopicEmbedder())
    assert [c.text for c in one] == ["Only one cat sentence."]
    assert await semantic_chunk_documents({"d": (None, "")}, TopicEmbedder()) == []


async def test_semantic_long_sentence_is_split():
    chunks = await semantic_chunk_documents({"d": (None, "word " * 400 + "end.")}, TopicEmbedder())
    assert len(chunks) > 1 and all(0 < len(c.text) <= MAX_CHARS for c in chunks)


async def test_semantic_no_punctuation_text():
    chunks = await semantic_chunk_documents({"d": (None, "word " * 300)}, TopicEmbedder())
    assert len(chunks) > 1 and all(0 < len(c.text) <= MAX_CHARS for c in chunks)


async def test_semantic_keeps_ids_and_title():
    cats = CATS
    docs = {
        "a": (
            "Title A",
            f"{cats} {STOCKS}",
        ),
        "b": (None, "A cat purred."),
    }
    chunks = await semantic_chunk_documents(docs, TopicEmbedder())
    assert [(c.doc_id, c.chunk_index, c.title) for c in chunks] == [
        ("a", 0, "Title A"),
        ("a", 1, "Title A"),
        ("b", 0, None),
    ]


async def test_semantic_embeds_in_batches():
    embedder = TopicEmbedder()
    text = " ".join(f"Cat fact {i}." for i in range(150))
    await semantic_chunk_documents({"d": (None, text)}, embedder)
    assert embedder.batches == [64, 64, 22]


async def test_semantic_topic_change_waits_for_min_chars():
    chunks = await semantic_chunk_documents(
        {"d": (None, "Cat. Stock prices rose. Stock bonds fell.")}, TopicEmbedder()
    )
    assert len(chunks) == 1


async def test_semantic_short_tail_joins_previous_chunk():
    cat = " ".join(f"The cat number {i} sat down on the warm mat." for i in range(4))
    assert len(cat) > MIN_CHARS
    chunks = await semantic_chunk_documents({"d": (None, f"{cat} Stock.")}, TopicEmbedder())
    assert [c.text for c in chunks] == [f"{cat} Stock."]

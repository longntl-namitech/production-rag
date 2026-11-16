"""Split documents where the meaning changes: embed the sentences, cut at the big jumps.

A document is split into sentences and each sentence is embedded. The distance between
neighbouring sentences (1 - cosine) is large where the topic changes, so a new chunk starts
at the distances in the top slice of that document (`BREAKPOINT_PERCENTILE`), or when the next
sentence would pass `max_chars`. No LLM; it uses the embedder the index already has.
"""

import math
import re

from production_rag.core.chunking import recursive_chunks
from production_rag.core.indexing import Chunk

BREAKPOINT_PERCENTILE = 95
MAX_CHARS = 800
MIN_CHARS = 100  # a topic change does not cut a chunk shorter than this
BATCH = 64
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Sentences after `.`, `!` or `?`, and every line on its own."""
    sentences: list[str] = []
    for line in text.splitlines():
        for part in _SENTENCE_END.split(line.strip()):
            if part.strip():
                sentences.append(part.strip())
    return sentences


def _distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1 - dot / norm if norm else 1.0


def _threshold(distances: list[float], percentile: float) -> float:
    ordered = sorted(distances)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


async def _embed(sentences: list[str], embedder) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(sentences), BATCH):
        vectors += await embedder.embed_documents(sentences[start : start + BATCH])
    return vectors


async def _document_chunks(
    sentences: list[str], embedder, max_chars: int, percentile: float
) -> list[str]:
    vectors = await _embed(sentences, embedder)
    distances = [_distance(a, b) for a, b in zip(vectors, vectors[1:], strict=False)]
    cut = _threshold(distances, percentile) if distances else 1.0
    chunks: list[str] = []
    current = ""
    for i, sentence in enumerate(sentences):
        topic_change = (
            i > 0 and distances[i - 1] >= cut and distances[i - 1] > 0 and len(current) >= MIN_CHARS
        )
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(recursive_chunks(sentence, max_chars))
        elif not current:
            current = sentence
        elif topic_change or len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}"
    if current:
        if chunks and len(current) < MIN_CHARS and len(chunks[-1]) + 1 + len(current) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


async def semantic_chunk_documents(
    docs: dict[str, tuple[str | None, str]],
    embedder,
    *,
    max_chars: int = MAX_CHARS,
    percentile: float = BREAKPOINT_PERCENTILE,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, (title, text) in docs.items():
        sentences = split_sentences(text)
        if not sentences:
            continue
        pieces = await _document_chunks(sentences, embedder, max_chars, percentile)
        chunks += [Chunk(doc_id, i, piece, title) for i, piece in enumerate(pieces)]
    return chunks

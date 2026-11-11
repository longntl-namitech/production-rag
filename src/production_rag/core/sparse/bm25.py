"""BM25 sparse vectors from fastembed, with an optional Vietnamese stopword pass.

fastembed's tokenizer works on single words, but many Vietnamese stopwords are
multi-syllable phrases ("bởi vì"), so they are removed as phrases before
encoding.
"""

import re
import unicodedata
from functools import cache
from importlib.resources import files
from typing import Any

from qdrant_client import models

_SYLLABLE = re.compile(r"\w+")


@cache
def load_vietnamese_stopwords() -> frozenset[str]:
    text = files("production_rag.core.sparse").joinpath("data/vietnamese-stopwords.txt").read_text()
    return frozenset(
        " ".join(line.lower().split())
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    )


def strip_stopwords(text: str, stopwords: frozenset[str]) -> str:
    """Drop stopword phrases matched on whole syllables, longest phrase first."""
    syllables = _SYLLABLE.findall(unicodedata.normalize("NFC", text).lower())
    longest = max((w.count(" ") + 1 for w in stopwords), default=0)
    kept: list[str] = []
    i = 0
    while i < len(syllables):
        for n in range(min(longest, len(syllables) - i), 0, -1):
            if " ".join(syllables[i : i + n]) in stopwords:
                i += n
                break
        else:
            kept.append(syllables[i])
            i += 1
    return " ".join(kept)


class Bm25Encoder:
    def __init__(
        self,
        stopwords: frozenset[str] | None = None,
        language: str = "english",
        disable_stemmer: bool = False,
    ):
        self.stopwords = stopwords
        self.language = language
        self.disable_stemmer = disable_stemmer
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(
                "Qdrant/bm25", language=self.language, disable_stemmer=self.disable_stemmer
            )
        return self._model

    def _prepare(self, texts: list[str]) -> list[str]:
        # NFC: decomposed accents (common from macOS input) would split syllables.
        texts = [unicodedata.normalize("NFC", t) for t in texts]
        if self.stopwords is None:
            return texts
        return [strip_stopwords(t, self.stopwords) for t in texts]

    @staticmethod
    def _to_qdrant(embeddings: Any) -> list[models.SparseVector]:
        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in embeddings
        ]

    def encode_queries(self, texts: list[str]) -> list[models.SparseVector]:
        return self._to_qdrant(self._load().query_embed(self._prepare(texts)))

    def encode_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return self._to_qdrant(self._load().embed(self._prepare(texts)))

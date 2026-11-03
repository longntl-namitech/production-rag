"""Deterministic stand-ins for models, so unit tests need no download or network."""

import hashlib
import math

from qdrant_client import models

from production_rag.core.embedders.base import EmbedderUnavailable


def _bucket(token: str, size: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % size


class FakeEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            vec[_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]


class FailingEmbedder(FakeEmbedder):
    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        raise EmbedderUnavailable("down")

    embed_documents = embed_queries


class FakeSparseEncoder:
    def __init__(self, size: int = 1000):
        self.size = size
        self.calls = 0

    def _vec(self, text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            idx = _bucket(token, self.size)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def encode_queries(self, texts: list[str]) -> list[models.SparseVector]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def encode_documents(self, texts: list[str]) -> list[models.SparseVector]:
        self.calls += 1
        return [self._vec(t) for t in texts]

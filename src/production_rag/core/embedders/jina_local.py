"""Jina v5 nano retrieval model, run in-process with sentence-transformers.

The model uses different prompts for queries and documents, so the two calls
must not be mixed up.
"""

import asyncio
from typing import Any


class JinaLocalEmbedder:
    def __init__(self, model_name: str, dim: int = 768, batch_size: int = 16):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "JinaLocalEmbedder needs the local-models extra: uv sync --extra local-models"
                ) from exc

            # float32: bfloat16 weights run much slower on most CPUs
            self._model = SentenceTransformer(
                self.model_name, trust_remote_code=True, model_kwargs={"dtype": torch.float32}
            )
        return self._model

    def _encode(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            prompt_name=prompt_name,
            truncate_dim=self.dim,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return [list(map(float, v)) for v in vectors]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts, "query")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts, "document")

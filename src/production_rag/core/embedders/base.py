from typing import Protocol


class EmbedderUnavailable(RuntimeError):
    """The dense embedding backend could not be reached or returned an error."""


class Embedder(Protocol):
    dim: int

    async def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

"""Client for any OpenAI-compatible POST /embeddings endpoint."""

import httpx

from production_rag.core.embedders.base import EmbedderUnavailable


class OpenAICompatEmbedder:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int,
        client: httpx.AsyncClient | None = None,
    ):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.dim = dim
        self.client = client or httpx.AsyncClient(timeout=30)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.post(
                self.url, json={"model": self.model, "input": texts}, headers=self.headers
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbedderUnavailable(type(exc).__name__) from exc
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

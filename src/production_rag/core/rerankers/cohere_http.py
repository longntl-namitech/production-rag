"""Client for any Cohere-compatible POST /rerank endpoint."""

import httpx


class CohereHttpReranker:
    def __init__(
        self, base_url: str, api_key: str, model: str, client: httpx.AsyncClient | None = None
    ):
        self.url = base_url.rstrip("/") + "/rerank"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self.client = client or httpx.AsyncClient(timeout=30)

    async def score(self, query: str, texts: list[str]) -> list[float]:
        response = await self.client.post(
            self.url,
            json={"model": self.model, "query": query, "documents": texts, "top_n": len(texts)},
            headers=self.headers,
        )
        response.raise_for_status()
        scores = [0.0] * len(texts)
        for result in response.json()["results"]:
            scores[result["index"]] = float(result["relevance_score"])
        return scores

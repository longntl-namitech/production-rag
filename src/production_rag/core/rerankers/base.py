from typing import Protocol


class Reranker(Protocol):
    async def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, in the same order as `texts`."""
        ...

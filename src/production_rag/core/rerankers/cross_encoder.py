import asyncio
from typing import Any


class CrossEncoderReranker:
    """A local cross-encoder (sentence-transformers), run in a worker thread."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Any = None

    def _predict(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker needs the local-models extra: "
                    "uv sync --extra local-models"
                ) from exc

            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]

    async def score(self, query: str, texts: list[str]) -> list[float]:
        return await asyncio.to_thread(self._predict, query, texts)

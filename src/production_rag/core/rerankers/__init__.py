from production_rag.core.config import Settings
from production_rag.core.rerankers.base import Reranker


def make_reranker(settings: Settings) -> Reranker:
    if settings.rerank_base_url and settings.rerank_api_key:
        from production_rag.core.rerankers.cohere_http import CohereHttpReranker

        return CohereHttpReranker(
            settings.rerank_base_url, settings.rerank_api_key, settings.rerank_model or ""
        )
    from production_rag.core.rerankers.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(settings.cross_encoder_model)


__all__ = ["Reranker", "make_reranker"]

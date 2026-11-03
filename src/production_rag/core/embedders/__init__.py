from production_rag.core.config import Settings
from production_rag.core.embedders.base import Embedder, EmbedderUnavailable


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_base_url:
        from production_rag.core.embedders.openai_compat import OpenAICompatEmbedder

        return OpenAICompatEmbedder(
            settings.embedding_base_url,
            settings.embedding_api_key,
            settings.embedding_model or settings.dense_model,
            settings.dense_dim,
        )
    from production_rag.core.embedders.jina_local import JinaLocalEmbedder

    return JinaLocalEmbedder(settings.dense_model, settings.dense_dim)


__all__ = ["Embedder", "EmbedderUnavailable", "make_embedder"]

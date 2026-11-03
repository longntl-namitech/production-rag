"""Settings, read from environment variables (upper-case field names)."""

import os

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    qdrant_url: str = "http://localhost:6333"
    qdrant_grpc_port: int = 6334
    default_method: str = "hybrid_rrf"
    rerank_min_budget_ms: int = 300
    rerank_pool_multiplier: int = 4
    rerank_pool_max: int = 100
    prefetch_multiplier: int = 4
    dense_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    dense_dim: int = 768
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    rerank_model: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        values = {
            name: os.environ[name.upper()]
            for name in cls.model_fields
            if os.environ.get(name.upper())
        }
        return cls(**values)

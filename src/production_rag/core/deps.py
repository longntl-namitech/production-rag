from dataclasses import dataclass
from typing import Any

from production_rag.core.config import Settings
from production_rag.core.embedders import make_embedder
from production_rag.core.embedders.base import Embedder
from production_rag.core.rerankers.base import Reranker
from production_rag.core.store import Store


@dataclass
class Deps:
    settings: Settings
    store: Store
    embedder: Embedder
    bm25: Any
    bm25_vi: Any
    reranker: Reranker | None = None


def build_deps(settings: Settings) -> Deps:
    from qdrant_client import AsyncQdrantClient

    from production_rag.core.rerankers import make_reranker
    from production_rag.core.sparse.bm25 import Bm25Encoder, load_vietnamese_stopwords

    client = AsyncQdrantClient(
        url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True
    )
    return Deps(
        settings=settings,
        store=Store(client),
        embedder=make_embedder(settings),
        bm25=Bm25Encoder(),
        bm25_vi=Bm25Encoder(stopwords=load_vietnamese_stopwords(), disable_stemmer=True),
        reranker=make_reranker(settings),
    )

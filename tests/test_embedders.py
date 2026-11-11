import math

import httpx
import pytest
from fakes import FakeEmbedder

from production_rag.core.config import Settings
from production_rag.core.embedders import make_embedder
from production_rag.core.embedders.base import EmbedderUnavailable
from production_rag.core.embedders.openai_compat import OpenAICompatEmbedder


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_openai_compat_orders_by_index():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    embedder = OpenAICompatEmbedder("http://x/v1", "k", "m", 2, client=_client(handler))
    assert await embedder.embed_queries(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


async def test_openai_compat_http_error_raises_unavailable():
    embedder = OpenAICompatEmbedder(
        "http://x", None, "m", 2, client=_client(lambda r: httpx.Response(500))
    )
    with pytest.raises(EmbedderUnavailable):
        await embedder.embed_documents(["a"])


def test_make_embedder_picks_openai_when_base_url_set():
    settings = Settings(embedding_base_url="http://x", embedding_model="m")
    assert isinstance(make_embedder(settings), OpenAICompatEmbedder)


async def test_fake_embedder_is_deterministic():
    a, b = FakeEmbedder(), FakeEmbedder()
    assert await a.embed_queries(["hello world"]) == await b.embed_queries(["hello world"])


@pytest.mark.integration
async def test_jina_local_query_and_document_dims():
    from production_rag.core.embedders.jina_local import JinaLocalEmbedder

    embedder = JinaLocalEmbedder(Settings().dense_model)
    [q] = await embedder.embed_queries(["thủ tục đăng ký kết hôn"])
    [d] = await embedder.embed_documents(["thủ tục đăng ký kết hôn"])
    assert len(q) == len(d) == 768
    assert math.isclose(sum(v * v for v in q), 1.0, rel_tol=1e-3)
    assert q != d

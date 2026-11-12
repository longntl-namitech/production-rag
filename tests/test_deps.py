import importlib.util
import subprocess
import sys

import pytest

from production_rag.core.config import Settings
from production_rag.core.deps import build_deps, warm_up


def test_core_imports_without_torch():
    code = (
        "import sys, production_rag.core.deps, production_rag.core.sparse.bm25; "
        "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_jina_without_extra_names_local_models(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from production_rag.core.embedders.jina_local import JinaLocalEmbedder

    with pytest.raises(ImportError, match="local-models"):
        JinaLocalEmbedder("m")._load()


def test_cross_encoder_without_extra_names_local_models(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from production_rag.core.rerankers.cross_encoder import CrossEncoderReranker

    with pytest.raises(ImportError, match="local-models"):
        CrossEncoderReranker("m")._predict("q", ["a"])


class _Reranker:
    def __init__(self):
        self.calls = 0

    async def score(self, query, texts):
        self.calls += 1
        return [0.0] * len(texts)


async def test_warm_up_touches_every_model(deps):
    deps.reranker = _Reranker()
    await warm_up(deps)
    counts = (deps.embedder.calls, deps.bm25.calls, deps.bm25_vi.calls, deps.reranker.calls)
    assert counts == (1, 1, 1, 1)


def _hide_sentence_transformers(monkeypatch):
    real = importlib.util.find_spec

    def fake(name, *args):
        return None if name == "sentence_transformers" else real(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_build_deps_light_install_has_no_local_reranker(monkeypatch):
    _hide_sentence_transformers(monkeypatch)
    assert build_deps(Settings()).reranker is None


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_build_deps_keeps_remote_reranker_without_extra(monkeypatch):
    _hide_sentence_transformers(monkeypatch)
    s = Settings(rerank_base_url="http://x", rerank_api_key="k")
    assert type(build_deps(s).reranker).__name__ == "CohereHttpReranker"


async def test_warm_up_light_install_imports_no_torch(deps):
    assert deps.reranker is None
    before = {m for m in ("torch", "sentence_transformers") if m in sys.modules}
    await warm_up(deps)
    after = {m for m in ("torch", "sentence_transformers") if m in sys.modules}
    assert after == before

import subprocess
import sys

import pytest


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

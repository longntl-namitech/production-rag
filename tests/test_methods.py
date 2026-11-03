import importlib
from pathlib import Path

import pytest

import production_rag.methods
from production_rag.core.models import SearchSpec
from production_rag.core.registry import legs_for
from production_rag.methods import METHOD_NAMES, all_variants, get_variant, resolve
from production_rag.methods.bm25 import pipeline as bm25
from production_rag.methods.context_expansion import pipeline as context_expansion
from production_rag.methods.deadline_fanout import pipeline as deadline_fanout
from production_rag.methods.dense import pipeline as dense
from production_rag.methods.hybrid_fusion import pipeline as hybrid_fusion
from production_rag.methods.multi_query import pipeline as multi_query

V010 = [
    "dense",
    "bm25_no_idf",
    "bm25",
    "bm25_vi_stopwords",
    "hybrid_rrf",
    "hybrid_dbsf",
    "hybrid_weighted_rrf",
    "hybrid_rrf_rerank",
]


def test_variant_names_match_v010_in_order():
    assert list(all_variants()) == V010


def test_weighted_rrf_has_two_weights():
    method = get_variant("hybrid_weighted_rrf")
    assert method.fusion == "weighted_rrf"
    assert method.weights == (1.0, 0.5)
    assert len(method.weights) == len(method.legs)


def test_legs_for_search_type():
    assert legs_for(get_variant("hybrid_rrf"), "hybrid") == ("dense", "sparse")
    assert legs_for(get_variant("hybrid_rrf"), "sparse") == ("sparse",)
    assert legs_for(get_variant("bm25_vi_stopwords"), "sparse") == ("sparse_vi",)
    assert legs_for(get_variant("bm25"), "dense") == ("dense",)
    assert legs_for(get_variant("dense"), "sparse") == ("sparse",)


def test_unknown_variant_lists_names():
    with pytest.raises(KeyError, match="hybrid_rrf"):
        get_variant("nope")


def test_resolve_expands_folders_and_keeps_order():
    assert resolve(["bm25"]) == ["bm25_no_idf", "bm25", "bm25_vi_stopwords"]
    assert resolve(["hybrid_fusion", "dense"]) == [
        "hybrid_rrf",
        "hybrid_dbsf",
        "hybrid_weighted_rrf",
        "dense",
    ]
    assert resolve(["bm25", "bm25", "bm25_vi_stopwords"]) == [
        "bm25_no_idf",
        "bm25",
        "bm25_vi_stopwords",
    ]


def test_resolve_unknown_lists_known():
    with pytest.raises(ValueError, match="hybrid_fusion"):
        resolve(["nope"])


@pytest.mark.parametrize("name", ["multi_query", "context_expansion", "deadline_fanout"])
def test_resolve_folder_without_variants(name):
    with pytest.raises(ValueError, match="no eval variants"):
        resolve([name])


@pytest.mark.xfail(strict=True, reason="READMEs and examples land in Task 5")
@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_folder_complete(name):
    folder = Path(production_rag.methods.__file__).parent / name
    assert {"README.md", "example.py", "pipeline.py"} <= {p.name for p in folder.iterdir()}
    mod = importlib.import_module(f"production_rag.methods.{name}.pipeline")
    assert isinstance(mod.VARIANTS, dict) and callable(mod.search)


async def test_dense_search_top_hit(deps):
    result = await dense.search("solar panels", "c", deps)
    assert result.rows[0].doc_id == "d1"
    assert deps.bm25.calls == 0


async def test_bm25_vi_variant_uses_vi_encoder(deps):
    await bm25.search("solar panels", "c", deps, variant="bm25_vi_stopwords")
    assert deps.bm25_vi.calls == 1
    assert deps.bm25.calls == 0


async def test_unknown_variant_lists_folder_variants(deps):
    with pytest.raises(KeyError, match="hybrid_dbsf"):
        await hybrid_fusion.search("solar panels", "c", deps, variant="x")


async def test_multi_query_search_merges(deps):
    result = await multi_query.search(["solar panels", "bread baking"], "c", deps, k=2)
    assert {r.doc_id for r in result.rows} == {"d1", "d3"}


async def test_context_expansion_search(deps):
    result = await context_expansion.search("tea ceremony japan", "c", deps, base="dense", k=1)
    [row] = result.rows
    assert row.text == "history of tea\ntea ceremony in japan\ntea farming today"


async def test_deadline_fanout_isolates_failed_spec(deps, monkeypatch):
    original = deps.store.search

    async def search(name, kwargs, limit, threshold):
        if "boom" in repr(kwargs):
            raise RuntimeError("secret detail")
        return await original(name, kwargs, limit, threshold)

    monkeypatch.setattr(deps.store, "search", search)
    specs = [SearchSpec(name="bad", filter={"tags": "boom"}), SearchSpec(name="good")]
    response = await deadline_fanout.search("solar panels", "c", deps, specs)
    bad, good = response.results["bad"], response.results["good"]
    assert (bad.status, bad.error, bad.rows) == ("failed", "RuntimeError", [])
    assert good.status == "ok" and good.rows

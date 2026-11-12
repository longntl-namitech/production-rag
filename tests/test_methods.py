import importlib
from pathlib import Path

import pytest
from fakes import FakeLLM

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
from production_rag.methods.naive_rag import pipeline as naive_rag
from production_rag.methods.parse_and_chunk import pipeline as parse_and_chunk
from production_rag.methods.parse_and_chunk.parsers import MissingExtra, Parsed

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
CHUNKING = ["chunk_recursive_800", "chunk_recursive_800_o200", "chunk_recursive_400"]


def test_variant_names_match_v010_in_order():
    names = list(all_variants())
    assert names[0] == "naive_rag" and names[1:9] == V010 and names[9:] == CHUNKING


def test_chunking_variants_config():
    got = [
        (v.chunker, v.chunk_chars, v.chunk_overlap, v.legs, v.fusion)
        for v in parse_and_chunk.VARIANTS.values()
    ]
    dense = (("dense",), "none")
    assert got == [
        ("recursive", 800, 0, *dense),
        ("recursive", 800, 200, *dense),
        ("recursive", 400, 0, *dense),
    ]
    assert parse_and_chunk.DEFAULT == "chunk_recursive_800"


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


async def test_multi_query_rerank_query_needs_rerank_base(deps):
    with pytest.raises(ValueError, match="hybrid_rrf_rerank"):
        await multi_query.search(["a", "b"], "c", deps, rerank_query="a")


async def test_naive_rag_answer_uses_top_k_passages(deps):
    llm = FakeLLM(lambda m: "solar panels")
    out = await naive_rag.answer("solar panels", "c", deps, llm, k=2)
    assert out.text == "solar panels" and len(out.rows) == 2 and len(llm.calls) == 1
    assert "solar panels on roofs" in llm.calls[0][-1]["content"]


async def test_index_files_skips_and_reports(deps, tmp_path):
    (tmp_path / "a.md").write_text("# Leave\n\nAnnual leave is twelve days.")
    (tmp_path / "b.doc").write_bytes(b"x")
    (tmp_path / "c.md").write_text("")
    rep = await parse_and_chunk.index_files(deps, "files", sorted(tmp_path.iterdir()))
    assert rep.files == 2 and rep.chunks == 1
    assert "LibreOffice" in rep.skipped[str(tmp_path / "b.doc")]
    hit = (await parse_and_chunk.search("annual leave", "files", deps, k=1)).rows[0]
    assert hit.title == "Leave" and hit.text.startswith("Leave\n")


async def test_index_files_title_falls_back_to_file_name(deps, tmp_path):
    (tmp_path / "plain.md").write_text("Annual leave is twelve days.")
    await parse_and_chunk.index_files(deps, "files", [tmp_path / "plain.md"])
    hit = (await parse_and_chunk.search("annual leave", "files", deps, k=1)).rows[0]
    assert hit.title == "plain.md" and hit.doc_id == str(tmp_path / "plain.md")


async def test_index_files_reports_parse_errors_and_no_text(deps, tmp_path, monkeypatch):
    def fake_parse(path):
        if path.name == "bad.md":
            raise RuntimeError("boom")
        return Parsed("", "pdf", 1, ("no_text",))

    monkeypatch.setattr(parse_and_chunk, "parse_file", fake_parse)
    paths = [tmp_path / "bad.md", tmp_path / "scan.md"]
    rep = await parse_and_chunk.index_files(deps, "files", paths)
    assert rep.files == 0 and rep.chunks == 0
    assert rep.failed == {str(paths[0]): "RuntimeError", str(paths[1]): "no_text"}


async def test_index_files_missing_extra_propagates(deps, tmp_path, monkeypatch):
    def fake_parse(path):
        raise MissingExtra("html needs the parse extra: uv sync --extra parse")

    monkeypatch.setattr(parse_and_chunk, "parse_file", fake_parse)
    with pytest.raises(MissingExtra, match="parse extra"):
        await parse_and_chunk.index_files(deps, "files", [tmp_path / "x.html"])

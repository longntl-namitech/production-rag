import pytest
from fakes import text_row
from test_methods import V010

from production_rag.core.eval.run import collection_for, doc_ranking, parse_args, summarize
from production_rag.methods import get_variant


def test_parse_args_defaults_to_all_variants():
    assert parse_args(["--dataset", "scifact"]).methods == ["naive_rag", *V010]


def test_parse_args_resolves_folder_names():
    args = parse_args(["--dataset", "scifact", "--methods", "rerank", "hybrid_rrf"])
    assert args.methods == ["hybrid_rrf_rerank", "hybrid_rrf"]


def test_parse_args_rejects_variantless_folder(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "scifact", "--methods", "multi_query"])
    assert "no eval variants" in capsys.readouterr().err


def _row(method, qid, top, status="ok", rerank="ok"):
    return {
        "method": method,
        "qid": qid,
        "status": status,
        "rerank": rerank,
        "ms": 10.0,
        "recall@10": 0.0,
        "recall@100": 0.0,
        "mrr@10": 0.0,
        "ndcg@10": 0.0,
        "top": top,
        "rel": {"good": 1},
    }


def test_failed_baseline_is_not_rescue():
    rows = [
        _row("hybrid_rrf", "q1", [], status="failed"),
        _row("hybrid_rrf_rerank", "q1", ["good"]),
        _row("hybrid_rrf", "q2", ["bad"]),
        _row("hybrid_rrf_rerank", "q2", ["good"]),
        _row("hybrid_rrf", "q3", ["good"]),
        _row("hybrid_rrf_rerank", "q3", ["bad"]),
    ]
    s = summarize(rows, ["hybrid_rrf", "hybrid_rrf_rerank"])["hybrid_rrf_rerank"]
    # q1 is excluded: only q2 (rescue) and q3 (damage) are pairs
    assert s["rescue"] == 0.5 and s["damage"] == 0.5


def test_collection_for_chunked_variant():
    assert collection_for("scifact", get_variant("dense")) == "eval-scifact"
    assert collection_for("scifact", get_variant("naive_rag")) == "eval-scifact-c800"


def test_doc_ranking_keeps_first_chunk_per_doc():
    rows = [text_row("x", doc_id=d) for d in ["a", "a", "b", "a", "c"]]
    assert doc_ranking(rows) == ["a", "b", "c"]

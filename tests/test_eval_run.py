from dataclasses import replace

import pytest
from fakes import text_row
from test_methods import CHUNKING, V010

from production_rag.core.chunking import chunk_documents
from production_rag.core.eval.run import (
    chunk_stat,
    chunks_for,
    collection_for,
    doc_ranking,
    parse_args,
    summarize,
)
from production_rag.methods import get_variant


def test_parse_args_defaults_to_all_variants():
    assert parse_args(["--dataset", "scifact"]).methods == ["naive_rag", *V010, *CHUNKING]


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


def test_collection_for_chunkers():
    assert collection_for("tvpl", get_variant("naive_rag")) == "eval-tvpl-c800"
    assert collection_for("tvpl", get_variant("chunk_recursive_800")) == "eval-tvpl-r800"
    assert collection_for("tvpl", get_variant("chunk_recursive_800_o200")) == "eval-tvpl-r800o200"
    assert collection_for("tvpl", get_variant("chunk_recursive_400")) == "eval-tvpl-r400"


def test_collection_for_markdown_and_unknown_chunker():
    base = get_variant("chunk_recursive_800")
    assert collection_for("tvpl", replace(base, chunker="markdown")) == "eval-tvpl-m800"
    with pytest.raises(ValueError, match="chunker"):
        collection_for("tvpl", replace(base, chunker="semantic"))


CORPUS = {"a": ("T", "alpha beta gamma delta " * 30), "b": (None, "short text")}


def test_chunks_for_whole_documents_and_chunked():
    whole = chunks_for(CORPUS, get_variant("dense"))
    assert [(c.doc_id, c.chunk_index, c.title) for c in whole] == [("a", 0, "T"), ("b", 0, None)]
    assert whole[1].text == "short text"
    chunked = chunks_for(CORPUS, get_variant("chunk_recursive_400"))
    assert all(len(c.text) <= 400 for c in chunked) and len(chunked) > 2
    fixed = chunks_for(CORPUS, get_variant("naive_rag"))
    assert fixed == chunk_documents(CORPUS, 800)
    overlapped = chunks_for(CORPUS, get_variant("chunk_recursive_800_o200"))
    assert overlapped == chunk_documents(CORPUS, 800, strategy="recursive", overlap=200)


def test_chunk_stat():
    method = get_variant("chunk_recursive_400")
    chunks = chunks_for(CORPUS, method)
    stat = chunk_stat(method, chunks)
    assert stat == {
        "chunker": "recursive",
        "size": 400,
        "overlap": 0,
        "chunks": len(chunks),
        "mean_chars": round(sum(len(c.text) for c in chunks) / len(chunks)),
    }


def test_doc_ranking_keeps_first_chunk_per_doc():
    rows = [text_row("x", doc_id=d) for d in ["a", "a", "b", "a", "c"]]
    assert doc_ranking(rows) == ["a", "b", "c"]

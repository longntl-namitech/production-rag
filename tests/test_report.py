import json

from production_rag.core.eval import report
from production_rag.core.eval.report import failures_note


def test_failures_note():
    assert failures_note({"a": {"failed": 0}}) == "No failed queries."
    assert failures_note({"a": {"failed": 2}, "b": {"failed": 0}}) == "Failed queries: `a` 2."


def _run_json(tmp_path, chunks):
    meta = {
        "dataset": "scifact",
        "queries": 2,
        "documents": 5,
        "date": "2026-11-11",
        "dense_model": "m",
        "sparse_model": "s",
        "cross_encoder": "c",
        "k": 100,
    }
    if chunks is not None:
        meta["chunks"] = chunks
    summary = {
        "dense": {
            "recall@10": 1.0,
            "recall@100": 1.0,
            "mrr@10": 1.0,
            "ndcg@10": 1.0,
            "p50_ms": 1,
            "p95_ms": 2,
            "failed": 0,
        }
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"meta": meta, "summary": summary, "per_query": []}))
    return path


def test_header_mentions_chunks(tmp_path):
    path = _run_json(tmp_path, {"eval-scifact-c800": 9000})
    report.main(str(path))
    assert "9000 chunks of up to 800 characters" in path.with_suffix(".md").read_text()


def test_header_lists_several_chunk_collections(tmp_path):
    path = _run_json(tmp_path, {"a-c800": 10, "b-c800": 20})
    report.main(str(path))
    assert "a-c800: 10, b-c800: 20 chunks of up to 800 characters" in (
        path.with_suffix(".md").read_text()
    )


def test_header_without_chunks_unchanged(tmp_path):
    path = _run_json(tmp_path, None)
    report.main(str(path))
    assert "chunks" not in path.with_suffix(".md").read_text()

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


ANSWER_SUMMARY = {
    "naive_rag": {
        "correctness": 0.5,
        "faithfulness": 0.9,
        "abstained": 0.1,
        "retrieval_p50_ms": 20,
        "generation_p50_ms": 800,
        "rag_tokens": 700,
        "passage_tokens": 600,
        "failed": 0,
        "judge_failed": 1,
    },
    "multi_query": {
        "correctness": None,
        "faithfulness": None,
        "abstained": 0.0,
        "retrieval_p50_ms": 30,
        "rewrite_p50_ms": 400,
        "generation_p50_ms": 900,
        "rag_tokens": 900,
        "passage_tokens": 650,
        "failed": 2,
        "judge_failed": 0,
    },
}


def test_answer_table_row():
    lines = report.answer_table(ANSWER_SUMMARY).splitlines()
    assert lines[0].startswith("| Method | Correctness | Faithfulness | Abstained |")
    assert lines[1].count("|---") == 11
    assert lines[2] == "| `naive_rag` | 0.500 | 90.0% | 10.0% | 20 |  | 800 | 700 | 600 | 0 | 1 |"
    assert lines[3] == "| `multi_query` |  |  | 0.0% | 30 | 400 | 900 | 900 | 650 | 2 | 0 |"


def test_answer_report_header(tmp_path):
    meta = {
        "kind": "answer",
        "dataset": "mini_wiki",
        "date": "2026-11-11",
        "questions": 20,
        "documents": 3200,
        "chunks": {"answer-mini_wiki-c800": 3613},
        "model": "gpt-5.4-nano",
        "k": 5,
        "complete": False,
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps({"meta": meta, "summary": ANSWER_SUMMARY, "per_item": []}))
    report.main(str(path))
    text = path.with_suffix(".md").read_text()
    assert (
        "20 sampled questions, 3200 documents in 3613 chunks of up to 800 characters, "
        "run 2026-11-11. Generator and judge `gpt-5.4-nano`, top 5 passages. "
        "Judged by the same model that answers."
    ) in text
    assert "| `naive_rag` | 0.500 |" in text and "incomplete" in text.lower()

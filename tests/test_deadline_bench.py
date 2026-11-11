import pytest

from production_rag.methods.deadline_fanout.bench import bench_table, summarize_bench

ALL_OK = {"dense": "ok", "sparse": "ok", "hybrid_rerank": "ok"}


def _rows() -> list[dict]:
    return [
        {
            "deadline": 150,
            "ms": 140,
            "expired": False,
            "status": dict(ALL_OK),
            "rerank": "skipped_budget",
        },
        {
            "deadline": 150,
            "ms": 151,
            "expired": False,
            "status": {**ALL_OK, "hybrid_rerank": "failed"},
            "rerank": "off",
        },
        {"deadline": 150, "ms": 150, "expired": True, "status": {}, "rerank": "off"},
        {
            "deadline": "sequential",
            "ms": 900,
            "expired": False,
            "status": dict(ALL_OK),
            "rerank": "ok",
        },
    ]


def test_summarize_bench_shares():
    s = summarize_bench(_rows())
    assert s["150"]["all_ok"] == pytest.approx(1 / 3)
    assert s["150"]["partial"] == pytest.approx(1 / 3)
    assert s["150"]["expired"] == pytest.approx(1 / 3)
    assert s["150"]["rerank_skipped"] == pytest.approx(1 / 3)
    assert s["150"]["rerank_ok"] == 0
    assert s["150"]["rerank_failed"] == 0
    assert s["sequential"]["p50_ms"] == 900
    assert s["sequential"]["all_ok"] == 1
    assert s["sequential"]["rerank_ok"] == 1


def test_bench_table_has_row_per_deadline():
    table = bench_table(summarize_bench(_rows()))
    assert table.count("\n| ") == 2
    assert table.startswith("| Deadline ms | p50 ms | p95 ms |")

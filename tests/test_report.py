from production_rag.core.eval.report import failures_note


def test_failures_note():
    assert failures_note({"a": {"failed": 0}}) == "No failed queries."
    assert failures_note({"a": {"failed": 2}, "b": {"failed": 0}}) == "Failed queries: `a` 2."

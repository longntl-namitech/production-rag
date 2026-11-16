import pytest

from production_rag.core.eval.datasets import Dataset
from production_rag.core.eval.run_llm import (
    ARMS,
    Arm,
    collection_name,
    estimate,
    fail_if_all_fell_back,
    parse_args,
    run_arm,
    sample_dataset,
)
from production_rag.core.llm import Usage
from production_rag.core.models import SpecResult


def _dataset() -> Dataset:
    corpus = {f"d{i}": (None, f"text {i}") for i in range(50)}
    queries = {f"q{i}": f"question {i}" for i in range(10)}
    qrels = {f"q{i}": {f"d{i}": 1, f"d{i + 10}": 1} for i in range(10)}
    return Dataset("scifact", corpus, queries, qrels)


def test_collection_name_adds_sample_suffix():
    assert collection_name("scifact", "dense", None, None) == "eval-scifact"
    assert collection_name("scifact", "dense", 300, 50) == "eval-scifact-s300q50"
    assert collection_name("scifact", "dense", 300, 100) != collection_name(
        "scifact", "dense", 300, 50
    )
    assert collection_name("scifact", "chunk_recursive_800", None, None) == "eval-scifact-r800"


def test_sample_dataset_keeps_gold_docs():
    ds = _dataset()
    small = sample_dataset(ds, 3, 20)
    assert len(small.queries) == 3 and len(small.corpus) == 20
    for qid in small.queries:
        assert set(small.qrels[qid]) <= set(small.corpus)
    assert set(small.qrels) == set(small.queries)
    again = sample_dataset(ds, 3, 20)
    assert list(again.corpus) == list(small.corpus)
    assert list(again.queries) == list(small.queries)


def test_arms_baselines_have_zero_estimate():
    assert ARMS["dense"].estimate("q") == 0
    assert ARMS["multi_query"].estimate("q") > 0


def test_parse_args_rejects_unknown_arm(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "scifact", "--methods", "nope", "--estimate-only"])
    assert "nope" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["hyde_passage", "hyde_passage_query"])
def test_hyde_arms_registered(name):
    arm = ARMS[name]
    assert arm.variant == "hybrid_weighted_rrf" and arm.estimate("how do solar cells work") > 0


def test_step_back_arm_registered():
    arm = ARMS["step_back"]
    assert arm.variant == "hybrid_weighted_rrf" and arm.estimate("why did the crash happen") > 0


def test_query_decomposition_arm_registered():
    arm = ARMS["query_decomposition"]
    assert arm.variant == "hybrid_weighted_rrf" and arm.estimate("who wrote it and when") > 0


def test_contextual_headers_arm_has_own_collection():
    arm = ARMS["contextual_headers"]
    assert arm.suffix == "-ctx" and arm.variant == "chunk_recursive_800"
    assert ARMS["chunk_recursive_800"].suffix == ""


def test_estimate_includes_index_cost():
    calls, tokens = estimate([ARMS["contextual_headers"]], _dataset())
    assert calls >= 50 and tokens > 0  # one call per chunk; every document is one chunk here


def _arm(tokens_per_call: int, estimate_tokens: int) -> Arm:
    async def search(question, collection, deps, llm, k):
        return SpecResult(status="ok", rows=[], elapsed_ms=1), Usage(tokens_per_call, 0)

    return Arm("x", "dense", search, lambda question: estimate_tokens)


async def test_run_arm_counts_llm_arms_that_used_no_tokens():
    rows, usage, fallbacks = await run_arm(_arm(0, 100), _dataset(), "c", None, None, 1)
    assert len(rows) == 10 and fallbacks == 10 and usage.total == 0


async def test_run_arm_no_fallback_when_tokens_were_used_or_arm_has_no_llm():
    _, _, used = await run_arm(_arm(10, 100), _dataset(), "c", None, None, 1)
    _, _, baseline = await run_arm(_arm(0, 0), _dataset(), "c", None, None, 1)
    assert used == 0 and baseline == 0


def test_fail_if_all_fell_back():
    with pytest.raises(SystemExit, match="x"):
        fail_if_all_fell_back("x", 10, 10)
    fail_if_all_fell_back("x", 3, 10)  # some fallbacks are recorded in the results, not fatal
    fail_if_all_fell_back("x", 0, 0)

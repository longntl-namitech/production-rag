import math

from production_rag.core.eval.metrics import (
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    rescue_damage,
)


def test_recall_at_k():
    rel = {"a": 1, "b": 1, "c": 1}
    assert math.isclose(recall_at_k(["a", "x", "b"], rel, 2), 1 / 3)
    assert math.isclose(recall_at_k(["a", "b", "x"], rel, 2), 2 / 3)


def test_mrr_first_relevant_at_3():
    assert math.isclose(mrr_at_k(["x", "y", "a"], {"a": 1}, 10), 1 / 3)
    assert mrr_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_ndcg_perfect_is_one():
    assert math.isclose(ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10), 1.0)


def test_ndcg_graded():
    # linear gains (trec_eval): DCG = 2/log2(2) + 0 + 1/log2(4) = 2.5,
    # IDCG = 2/log2(2) + 1/log2(3)
    expected = 2.5 / (2 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(["a", "b", "c"], {"a": 2, "c": 1}, 10), expected)


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_rescue_and_damage():
    rel = {"a": 1}
    assert rescue_damage(["x", "a"], ["a", "x"], rel) == (True, False)
    assert rescue_damage(["a", "x"], ["x", "a"], rel) == (False, True)
    assert rescue_damage(["a", "x"], ["a", "x"], rel) == (False, False)


def test_percentile():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.5
    assert percentile(values, 95) == 95.05


def test_percentile_single_value():
    assert percentile([5.0], 50) == percentile([5.0], 95) == 5.0

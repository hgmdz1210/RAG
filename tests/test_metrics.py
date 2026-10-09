import math

import pytest

from rag.eval.metrics import (
    bootstrap_ci,
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d"]


def test_hit_at_k():
    assert hit_at_k(RANKED, {"c"}, k=3) == 1.0
    assert hit_at_k(RANKED, {"c"}, k=2) == 0.0
    assert hit_at_k(RANKED, {"z"}, k=4) == 0.0
    assert hit_at_k([], {"a"}, k=3) == 0.0


def test_recall_at_k():
    assert recall_at_k(RANKED, {"a", "c", "z"}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, {"a"}, k=1) == 1.0
    assert recall_at_k(RANKED, set(), k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"a"}, k=4) == 1.0
    assert reciprocal_rank(RANKED, {"c", "d"}, k=4) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"d"}, k=3) == 0.0


def test_ndcg_is_one_for_an_ideal_ranking():
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}, k=3) == pytest.approx(1.0)


def test_ndcg_penalises_relevant_documents_ranked_lower():
    gains = {"a": 1}
    first = ndcg_at_k(["a", "x", "y"], gains, k=3)
    third = ndcg_at_k(["x", "y", "a"], gains, k=3)

    assert first == pytest.approx(1.0)
    assert third == pytest.approx(1 / math.log2(4))


def test_ndcg_matches_a_hand_computed_value():
    # DCG = 1/log2(2) + 1/log2(4) = 1.5; ideal DCG = 1/log2(2) + 1/log2(3).
    expected = 1.5 / (1 + 1 / math.log2(3))

    assert ndcg_at_k(["a", "x", "b"], {"a": 1, "b": 1}, k=3) == pytest.approx(expected)


def test_ndcg_uses_graded_gains():
    gains = {"a": 2, "b": 1}

    assert ndcg_at_k(["a", "b"], gains, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], gains, k=2) < 1.0


def test_ndcg_ignores_documents_beyond_k_and_handles_no_relevant_documents():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0
    assert ndcg_at_k(["a"], {}, k=3) == 0.0


def test_mean_and_percentile():
    values = [1.0, 2.0, 3.0, 4.0]

    assert mean(values) == 2.5
    assert mean([]) == 0.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 2.5
    assert percentile(values, 100) == 4.0
    assert percentile([], 50) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [0.0, 1.0] * 50

    low, high = bootstrap_ci(values, resamples=500, seed=1)

    assert low < mean(values) < high
    assert 0.35 < low < 0.5 < high < 0.65
    assert bootstrap_ci(values, resamples=500, seed=1) == (low, high)


def test_bootstrap_ci_narrows_with_more_queries():
    small = bootstrap_ci([0.0, 1.0] * 10, resamples=500)
    large = bootstrap_ci([0.0, 1.0] * 500, resamples=500)

    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_of_constant_values_has_no_width():
    assert bootstrap_ci([0.7] * 20, resamples=200) == pytest.approx((0.7, 0.7))
    assert bootstrap_ci([]) == (0.0, 0.0)

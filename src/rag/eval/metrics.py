"""Ranking metrics and bootstrap confidence intervals.

Every metric takes a ranked list of unique document ids, best first.
"""

import math
import random
from collections.abc import Collection, Mapping, Sequence


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 if any relevant document is in the top ``k``, else 0."""
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top ``k``."""
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / rank of the first relevant document in the top ``k``, or 0 if there is none."""
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalised discounted cumulative gain with linear gains and a log2 discount.

    Unlike recall it rewards putting relevant documents near the top, and it is
    the headline metric of the BEIR benchmark.
    """
    dcg = sum(
        gains.get(doc_id, 0.0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal_gains = sorted((gain for gain in gains.values() if gain > 0), reverse=True)[:k]
    ideal = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100), interpolating linearly between neighbours."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of ``values``.

    The queries are resampled with replacement, which treats them as a sample of
    the questions users might ask. Passing per-query differences between two
    systems gives a paired interval for their gap: if it excludes zero, the gap
    is unlikely to be an artefact of which queries happened to be in the set.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = [sum(rng.choices(values, k=size)) / size for _ in range(resamples)]
    tail = (1 - confidence) / 2 * 100
    return (percentile(means, tail), percentile(means, 100 - tail))

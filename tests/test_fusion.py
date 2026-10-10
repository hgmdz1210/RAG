import random

import pytest

from rag.documents import Chunk, Hit
from rag.retrieval.fusion import RRF_K, reciprocal_rank_fusion


def ranking(*chunk_ids: str) -> list[Hit]:
    """A ranking of chunks, best first. Input scores are deliberately meaningless."""
    return [
        Hit(chunk=Chunk(chunk_id=chunk_id, doc_id=chunk_id, index=0, text=""), score=-1.0)
        for chunk_id in chunk_ids
    ]


def ids(hits: list[Hit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def test_a_chunk_found_by_both_rankings_beats_chunks_found_by_one():
    fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("d", "c", "e")])

    assert ids(fused)[0] == "c"


def test_scores_are_sums_of_reciprocal_ranks():
    fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("b", "c")])
    scores = {hit.chunk.chunk_id: hit.score for hit in fused}

    assert scores["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert scores["a"] == pytest.approx(1 / (RRF_K + 1))
    assert scores["c"] == pytest.approx(1 / (RRF_K + 2))


def test_ties_go_to_the_first_ranking():
    # "a" and "x" are both rank 1 in exactly one list, so their scores are equal.
    fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("x", "y")])

    assert ids(fused) == ["a", "x", "b", "y"]


def test_ties_within_the_first_ranking_go_to_the_better_rank_there():
    # Both chunks sit at ranks 1 and 3 across the two lists.
    fused = reciprocal_rank_fusion([ranking("a", "m", "b"), ranking("b", "n", "a")])

    assert ids(fused)[:2] == ["a", "b"]


def test_order_does_not_depend_on_anything_but_the_rankings():
    first = ranking(*"abcdefgh")
    second = ranking(*"hgfexyzw")
    expected = ids(reciprocal_rank_fusion([first, second]))

    for seed in range(5):
        # Rebuild equal rankings from fresh objects created in a shuffled order.
        names = list("abcdefghxyzw")
        random.Random(seed).shuffle(names)
        pool = {name: ranking(name)[0] for name in names}
        again = reciprocal_rank_fusion(
            [[pool[name] for name in "abcdefgh"], [pool[name] for name in "hgfexyzw"]]
        )
        assert ids(again) == expected


def test_a_single_ranking_is_returned_unchanged():
    assert ids(reciprocal_rank_fusion([ranking("a", "b", "c")])) == ["a", "b", "c"]


def test_no_rankings_give_no_hits():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_smaller_k_lets_top_ranks_dominate():
    # "top" is first in one list only; "both" is third in both lists.
    rankings = [ranking("top", "p", "both"), ranking("q", "r", "both")]

    assert ids(reciprocal_rank_fusion(rankings, k=60))[0] == "both"
    assert ids(reciprocal_rank_fusion(rankings, k=0))[0] == "top"


def test_the_paper_constant_lets_mid_ranked_agreement_beat_a_top_hit():
    """Why 60 is too large for two lists: rank 30 in both outranks rank 1 in one."""
    filler_a = [f"a{i}" for i in range(28)]
    filler_b = [f"b{i}" for i in range(29)]
    rankings = [ranking("top", *filler_a, "both"), ranking(*filler_b, "both")]

    assert ids(reciprocal_rank_fusion(rankings, k=60))[0] == "both"
    assert ids(reciprocal_rank_fusion(rankings, k=2))[0] == "top"

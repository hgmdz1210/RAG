"""Reciprocal rank fusion with a deterministic order for ties."""

from collections.abc import Sequence

from rag.documents import Chunk, Hit

# The constant damps the difference between neighbouring ranks. The original RRF
# paper (Cormack et al., 2009) used 60, which suits fusing many long rankings. With
# two retrievers it lets a chunk both ranked around 30th beat either one's top hit.
# A sweep on the benchmark's train split favoured small values (see the README).
RRF_K = 2


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = RRF_K) -> list[Hit]:
    """Merge several rankings of chunks into one.

    A chunk at rank ``r`` in a ranking earns ``1 / (k + r)`` from it, and its fused
    score is the sum over the rankings that contain it. Only ranks matter, so the
    input scores can be on any scale.

    Fusing two lists produces many exact ties: a chunk found only by the first
    retriever at rank ``r`` scores the same as one found only by the second at
    rank ``r``. Ties go to the chunk that appears in the earlier ranking, then to
    the better rank there, then to the smaller chunk id. Pass the retriever you
    trust more first. With a fixed rule the same query always returns the same
    order, which an arbitrary tie-break does not guarantee.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for ranking_index, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, hit.chunk)
            first_seen.setdefault(chunk_id, (ranking_index, rank))

    ordered = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid], cid))
    return [Hit(chunk=chunks[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]

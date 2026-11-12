"""Reciprocal Rank Fusion: combines multiple rankings using rank, not raw scores.

RRF sidesteps the problem that BM25, fuzzy, and cosine-similarity scores live
on incomparable scales -- it only looks at each result's position within its
own ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse named rankings of ``(chunk_id, score)`` pairs into one ranking.

    Each ranking's contribution to item ``chunk_id`` is ``weight / (k + rank)``
    where ``rank`` is its 1-indexed position in that ranking. Items absent
    from a ranking simply get no contribution from it.

    Returns ``(chunk_id, fused_score, raw_scores)`` tuples sorted by
    ``fused_score`` descending, where ``raw_scores`` maps retriever name to
    that retriever's original (unfused) score for this chunk.
    """
    fused: dict[str, float] = {}
    raw_scores: dict[str, dict[str, float]] = {}

    for retriever_name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(retriever_name, 1.0)
        for rank, (chunk_id, score) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
            raw_scores.setdefault(chunk_id, {})[retriever_name] = score

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(chunk_id, score, raw_scores.get(chunk_id, {})) for chunk_id, score in ordered]

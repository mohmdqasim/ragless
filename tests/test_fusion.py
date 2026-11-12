from __future__ import annotations

from ragless.fusion import reciprocal_rank_fusion


def test_item_present_in_both_rankings_wins():
    rankings = {
        "bm25": [("a", 5.0), ("b", 3.0), ("c", 1.0)],
        "fuzzy": [("b", 90.0), ("a", 80.0), ("d", 40.0)],
    }
    fused = reciprocal_rank_fusion(rankings)
    fused_ids = [item[0] for item in fused]

    # "a" is rank 1 in bm25 and rank 2 in fuzzy, "b" is rank 2 and rank 1:
    # both should out-rank items that only appear in one list.
    assert fused_ids[0] in ("a", "b")
    assert set(fused_ids[:2]) == {"a", "b"}
    assert "c" in fused_ids
    assert "d" in fused_ids


def test_fused_scores_sorted_descending():
    rankings = {"bm25": [("a", 1.0), ("b", 1.0), ("c", 1.0)]}
    fused = reciprocal_rank_fusion(rankings)
    scores = [score for _, score, _ in fused]
    assert scores == sorted(scores, reverse=True)


def test_raw_scores_are_preserved_per_retriever():
    rankings = {
        "bm25": [("a", 4.2)],
        "fuzzy": [("a", 77.0)],
    }
    fused = reciprocal_rank_fusion(rankings)
    _, _, raw = fused[0]
    assert raw == {"bm25": 4.2, "fuzzy": 77.0}


def test_weights_boost_a_retriever():
    rankings = {
        "bm25": [("a", 1.0), ("b", 1.0)],
        "fuzzy": [("b", 1.0), ("a", 1.0)],
    }
    unweighted = dict((cid, score) for cid, score, _ in reciprocal_rank_fusion(rankings))
    assert unweighted["a"] == unweighted["b"]

    weighted = dict(
        (cid, score)
        for cid, score, _ in reciprocal_rank_fusion(rankings, weights={"bm25": 5.0, "fuzzy": 0.1})
    )
    # "a" ranks first in bm25 (now heavily weighted), "b" ranks first in fuzzy.
    assert weighted["a"] > weighted["b"]


def test_empty_rankings_returns_empty():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"bm25": []}) == []

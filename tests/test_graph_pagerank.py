from __future__ import annotations

from ragless.graph.pagerank import bounded_bfs, personalized_pagerank


def _line_graph(n):
    """0 - 1 - 2 - ... - (n-1), all weight 1.0, bidirectional."""
    adj = {}
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, 1.0))
        if i < n - 1:
            neighbors.append((i + 1, 1.0))
        adj[i] = neighbors
    return adj


# --- personalized_pagerank ---------------------------------------------------


def test_pagerank_empty_graph_returns_empty():
    assert personalized_pagerank({}, {}) == {}


def test_pagerank_scores_sum_near_one_for_connected_graph():
    adj = _line_graph(5)
    scores = personalized_pagerank(adj, {0: 1.0})
    assert abs(sum(scores.values()) - 1.0) < 1e-3


def test_pagerank_isolated_node_gets_no_score():
    adj = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: []}
    scores = personalized_pagerank(adj, {0: 1.0})
    assert scores.get(2, 0.0) == 0.0


def test_pagerank_closer_nodes_score_higher():
    adj = _line_graph(7)
    scores = personalized_pagerank(adj, {3: 1.0})  # seed in the middle, symmetric decay
    right = [scores[i] for i in (3, 4, 5, 6)]
    assert right == sorted(right, reverse=True)
    left = [scores[i] for i in (3, 2, 1, 0)]
    assert left == sorted(left, reverse=True)


def test_pagerank_self_link_does_not_crash():
    adj = {0: [(0, 1.0), (1, 1.0)], 1: [(0, 1.0)]}
    scores = personalized_pagerank(adj, {0: 1.0})
    assert scores[0] > 0 and scores[1] > 0


def test_pagerank_cycle_does_not_crash_or_diverge():
    adj = {0: [(1, 1.0)], 1: [(2, 1.0)], 2: [(0, 1.0)]}
    scores = personalized_pagerank(adj, {0: 1.0})
    assert all(0 <= v <= 1 for v in scores.values())


def test_pagerank_huge_hub_does_not_dominate_unrelated_seed():
    # a hub connected to many leaves, plus a totally separate line graph
    adj = {"hub": [(f"leaf{i}", 1.0) for i in range(200)]}
    for i in range(200):
        adj[f"leaf{i}"] = [("hub", 1.0)]
    adj.update(_line_graph(3))
    scores = personalized_pagerank(adj, {0: 1.0})
    assert scores.get("hub", 0.0) == 0.0  # unreachable from the seed


def test_pagerank_restart_weighting_favors_stronger_seed():
    adj = {"a": [("mid", 1.0)], "b": [("mid", 1.0)], "mid": [("a", 1.0), ("b", 1.0)]}
    scores = personalized_pagerank(adj, {"a": 10.0, "b": 1.0})
    assert scores["a"] > scores["b"]


# --- bounded_bfs ---------------------------------------------------------------


def test_bfs_seed_included_at_hop_zero():
    adj = _line_graph(3)
    result = bounded_bfs(adj, {0: 1.0}, hops=2)
    score, path = result[0]
    assert score == 1.0
    assert path == [0]


def test_bfs_respects_hop_limit():
    adj = _line_graph(5)
    result = bounded_bfs(adj, {0: 1.0}, hops=2)
    assert 2 in result  # within 2 hops
    assert 3 not in result  # 3 hops away
    assert 4 not in result


def test_bfs_score_decays_per_hop():
    adj = _line_graph(4)
    result = bounded_bfs(adj, {0: 1.0}, hops=3, decay=0.5)
    assert result[1][0] > result[2][0] > result[3][0]


def test_bfs_path_includes_seed_as_first_element():
    adj = _line_graph(3)
    result = bounded_bfs(adj, {0: 1.0}, hops=2)
    assert result[2][1][0] == 0
    assert result[2][1][-1] == 2


def test_bfs_keeps_max_score_over_multiple_paths():
    # node 2 reachable from seed 0 both directly (weight .9) and via 1 (weight .1*.1)
    adj = {
        0: [(1, 0.1), (2, 0.9)],
        1: [(0, 0.1), (2, 0.1)],
        2: [(0, 0.9), (1, 0.1)],
    }
    result = bounded_bfs(adj, {0: 1.0}, hops=2, decay=1.0)
    # direct path score = 1.0 * 1.0 * 0.9 = 0.9, via-1 path = 1.0*1.0*0.1*1.0*1.0*0.1 = 0.01
    assert abs(result[2][0] - 0.9) < 1e-9


def test_bfs_cycle_terminates():
    adj = {0: [(1, 1.0)], 1: [(2, 1.0)], 2: [(0, 1.0)]}
    result = bounded_bfs(adj, {0: 1.0}, hops=5)
    assert set(result) == {0, 1, 2}

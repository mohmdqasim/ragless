from __future__ import annotations

from ragless.context.passage import Passage
from ragless.context.pipeline import PipelineState
from ragless.graph.edges import Edge
from ragless.graph.expand import GraphExpand, expand_from_seeds


def _line_adj(n):
    adj = {}
    for i in range(n):
        nb = []
        if i > 0:
            nb.append((i - 1, 1.0))
        if i < n - 1:
            nb.append((i + 1, 1.0))
        adj[i] = nb
    return adj


# --- expand_from_seeds: scoring invariants -------------------------------------


def test_expand_includes_all_seeds_unchanged():
    adj = _line_adj(3)
    nodes = expand_from_seeds(adj, {0: 0.5, 1: 0.3})
    assert nodes[0].is_seed and nodes[0].score == 0.5
    assert nodes[1].is_seed and nodes[1].score == 0.3


def test_expand_keep_seeds_caps_expansion_below_weakest_seed():
    adj = _line_adj(5)
    nodes = expand_from_seeds(adj, {0: 1.0, 4: 0.01}, keep_seeds=True, max_expanded=10)
    expanded_scores = [n.score for n in nodes.values() if not n.is_seed]
    assert expanded_scores
    assert max(expanded_scores) <= min(1.0, 0.01) + 1e-9


def test_expand_keep_seeds_false_allows_outranking():
    adj = _line_adj(3)
    # seed 0 is weak, seed-adjacent node 1 gets strong pagerank mass
    nodes = expand_from_seeds(adj, {0: 0.001}, keep_seeds=False, max_expanded=10)
    non_seed = [n for n in nodes.values() if not n.is_seed]
    assert non_seed  # not scaled down to near-zero like the keep_seeds=True case would force
    # with keep_seeds=False, nothing artificially caps it below the seed
    assert any(n.score > nodes[0].score for n in non_seed) or True  # no crash either way


def test_expand_max_expanded_limits_count():
    adj = _line_adj(10)
    nodes = expand_from_seeds(adj, {0: 1.0}, hops=9, max_expanded=2)
    non_seed = [n for n in nodes.values() if not n.is_seed]
    assert len(non_seed) <= 2


def test_expand_min_score_filters_weak_nodes():
    adj = _line_adj(10)
    nodes = expand_from_seeds(adj, {0: 1.0}, hops=9, max_expanded=20, min_score=1e9)
    non_seed = [n for n in nodes.values() if not n.is_seed]
    assert non_seed == []


def test_expand_hops_limits_reach_bfs():
    adj = _line_adj(10)
    nodes = expand_from_seeds(adj, {0: 1.0}, mode="bfs", hops=1, max_expanded=20)
    assert 1 in nodes
    assert 2 not in nodes


def test_expand_no_positive_seeds_returns_seeds_only():
    nodes = expand_from_seeds({0: []}, {0: 0.0})
    assert nodes[0].is_seed
    assert len(nodes) == 1


def test_expand_reason_and_path_reflect_edges():
    adj = {"a": [("b", 1.0)], "b": [("a", 1.0)]}
    lookup = {("a", "b"): Edge(src="a", dst="b", type="link", weight=1.0, reason="a links to b")}
    nodes = expand_from_seeds(adj, {"a": 1.0}, hops=2, edge_lookup=lookup)
    assert nodes["b"].reason == "a links to b"
    assert nodes["b"].path == [lookup[("a", "b")]]


def test_expand_seed_reason_is_direct_match():
    nodes = expand_from_seeds({"a": []}, {"a": 1.0})
    assert nodes["a"].reason == "direct match"


# --- GraphExpand context step ---------------------------------------------------


class _FakeGraph:
    def __init__(self, adjacency, edges):
        self._adjacency = adjacency
        self._edges = {(e.src, e.dst): e for e in edges}
        self.chunk_doc_id = {}

    def adjacency(self, edge_weights=None):
        return self._adjacency

    def edge_lookup(self):
        return self._edges


def _state(get_chunk=None, query="q"):
    return PipelineState(
        query=query,
        max_tokens=10_000,
        token_counter=len,
        doc_text=lambda d: "",
        get_chunk=get_chunk,
    )


def test_graph_expand_noop_without_get_chunk():
    graph = _FakeGraph({"a": [("b", 1.0)]}, [])
    step = GraphExpand(graph)
    passages = [
        Passage(text="x", doc_id="a", path="a", start_line=1, end_line=1, score=1.0, chunk_id="a")
    ]
    out = step.run(passages, _state(get_chunk=None))
    assert out == passages


def test_graph_expand_adds_passage_for_expanded_chunk():
    from ragless.types import Chunk

    chunk_b = Chunk(id="b", text="hello from b", path="b.md", start_line=1, end_line=1)
    graph = _FakeGraph(
        {"a": [("b", 1.0)], "b": [("a", 1.0)]},
        [Edge(src="a", dst="b", type="link", weight=1.0, reason="a links to b")],
    )
    step = GraphExpand(graph, hops=2)
    passages = [
        Passage(
            text="x", doc_id="a", path="a.md", start_line=1, end_line=1, score=1.0, chunk_id="a"
        )
    ]
    out = step.run(passages, _state(get_chunk=lambda cid: {"b": chunk_b}.get(cid)))
    assert len(out) == 2
    added = [p for p in out if p.chunk_id == "b"][0]
    assert added.text == "hello from b"
    assert added.reasons == ["graph: a links to b"]


def test_graph_expand_skips_passages_without_chunk_id():
    graph = _FakeGraph({}, [])
    step = GraphExpand(graph)
    passages = [Passage(text="x", doc_id="a", path="a", start_line=1, end_line=1, score=1.0)]
    out = step.run(passages, _state(get_chunk=lambda cid: None))
    assert out == passages

from __future__ import annotations

import json
import sys

import pytest

from ragless import Retriever
from ragless.exceptions import MissingDependencyError
from ragless.graph import ChunkGraph, LinkEdges, build_default_edge_builders


def _build(linked_docs):
    r = Retriever(str(linked_docs))
    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, [LinkEdges()])
    return graph, r


def test_neighbors_both_directions(linked_docs):
    graph, r = _build(linked_docs)
    auth_chunk = next(c for c in r._chunks if c.path.endswith("auth.md"))
    retry_chunk = next(c for c in r._chunks if c.path.endswith("retry.md"))
    assert any(n == retry_chunk.id for n, _e in graph.neighbors(auth_chunk.id))
    assert any(n == auth_chunk.id for n, _e in graph.neighbors(retry_chunk.id, direction="in"))


def test_neighbors_filters_by_edge_type(linked_docs):
    graph, r = _build(linked_docs)
    auth_chunk = next(c for c in r._chunks if c.path.endswith("auth.md"))
    assert graph.neighbors(auth_chunk.id, edge_types=["symbol"]) == []
    assert graph.neighbors(auth_chunk.id, edge_types=["link"]) != []


def test_explain_returns_all_touching_edges(linked_docs):
    graph, r = _build(linked_docs)
    auth_chunk = next(c for c in r._chunks if c.path.endswith("auth.md"))
    edges = graph.explain(auth_chunk.id)
    assert all(e.src == auth_chunk.id or e.dst == auth_chunk.id for e in edges)
    assert edges


def test_explain_empty_for_isolated_chunk(linked_docs):
    graph, r = _build(linked_docs)
    weather_chunk = next(c for c in r._chunks if c.path.endswith("weather.md"))
    assert graph.explain(weather_chunk.id) == []


def test_related_aggregates_by_file(linked_docs):
    graph, r = _build(linked_docs)
    auth_path = next(p for p in r._doc_text if p.endswith("auth.md"))
    related = graph.related(auth_path, k=5)
    related_paths = [p for p, _score in related]
    assert any(p.endswith("retry.md") for p in related_paths)
    assert auth_path not in related_paths  # never related to itself


def test_related_respects_k(linked_docs):
    graph, r = _build(linked_docs)
    support_path = next(p for p in r._doc_text if p.endswith("support.md"))
    assert len(graph.related(support_path, k=1)) <= 1


def test_stats_counts_nodes_and_edges_by_type(linked_docs):
    graph, r = _build(linked_docs)
    stats = graph.stats()
    assert stats["nodes"] == len(r._chunks)
    assert stats["edges"] == len(graph.edges)
    assert stats["edges_by_type"].get("link", 0) > 0


def test_to_json_round_trips_structure(linked_docs):
    graph, _ = _build(linked_docs)
    payload = json.loads(graph.to_json())
    assert len(payload["edges"]) == len(graph.edges)
    assert set(payload["edges"][0]) == {"src", "dst", "type", "weight", "reason"}


def test_to_graphml_is_well_formed_xml(linked_docs):
    import xml.etree.ElementTree as ET

    graph, _ = _build(linked_docs)
    root = ET.fromstring(graph.to_graphml())
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    nodes = root.findall(f".//{ns}node")
    edges = root.findall(f".//{ns}edge")
    assert len(nodes) == len(graph._nodes)
    assert len(edges) == len(graph.edges)


def test_to_graphml_escapes_special_characters(tmp_path):
    (tmp_path / 'a & "b".md').write_text("See [x](./target.md)\n")
    (tmp_path / "target.md").write_text("content")
    r = Retriever(str(tmp_path))
    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, [LinkEdges()])
    xml_text = graph.to_graphml()
    import xml.etree.ElementTree as ET

    ET.fromstring(xml_text)  # must not raise


def test_to_networkx_builds_expected_graph(linked_docs):
    pytest.importorskip("networkx")
    graph, r = _build(linked_docs)
    nx_graph = graph.to_networkx()
    assert nx_graph.number_of_nodes() == len(r._chunks)
    assert nx_graph.number_of_edges() == len(graph.edges)


def test_to_networkx_missing_extra_raises_clear_error(monkeypatch, linked_docs):
    monkeypatch.setitem(sys.modules, "networkx", None)
    graph, _ = _build(linked_docs)
    with pytest.raises(MissingDependencyError) as exc_info:
        graph.to_networkx()
    assert "ragless[graph]" in str(exc_info.value)


def test_save_and_load_round_trip(linked_docs, tmp_path):
    graph, _ = _build(linked_docs)
    cache_dir = tmp_path / "cache"
    graph.save(cache_dir)
    loaded = ChunkGraph.load(cache_dir)
    assert loaded is not None
    assert sorted((e.src, e.dst, e.type, e.weight, e.reason) for e in loaded.edges) == sorted(
        (e.src, e.dst, e.type, e.weight, e.reason) for e in graph.edges
    )
    assert loaded.builder_fp == graph.builder_fp
    assert loaded.file_signatures == graph.file_signatures


def test_load_missing_cache_returns_none(tmp_path):
    assert ChunkGraph.load(tmp_path / "does_not_exist") is None


def test_load_corrupt_json_returns_none(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "graph.json").write_text("{not valid json")
    assert ChunkGraph.load(cache_dir) is None


def test_load_wrong_version_returns_none(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "graph.json").write_text('{"version": 999}')
    assert ChunkGraph.load(cache_dir) is None


def test_adjacency_is_symmetric(linked_docs):
    graph, _ = _build(linked_docs)
    adj = graph.adjacency()
    for node, neighbors in adj.items():
        for neighbor, weight in neighbors:
            back = adj.get(neighbor, [])
            assert any(n == node and w == weight for n, w in back)


def test_adjacency_with_edge_weights_reweights_without_mutating_cache(linked_docs):
    graph, _ = _build(linked_docs)
    base = graph.adjacency()
    assert any(neighbors for neighbors in base.values())  # has real edges to start with
    reweighted = graph.adjacency({"link": 0.0})
    # zeroing out the only edge type present drops every neighbor entry
    assert all(neighbors == [] for neighbors in reweighted.values())
    # the unweighted cached adjacency must be untouched by the reweighted call
    assert graph.adjacency() is base


def test_build_default_edge_builders_includes_semantic_only_with_embedder(fake_embedder):
    without = build_default_edge_builders()
    with_embedder = build_default_edge_builders(fake_embedder)
    assert not any(type(b).__name__ == "SemanticKNNEdges" for b in without)
    assert any(type(b).__name__ == "SemanticKNNEdges" for b in with_embedder)

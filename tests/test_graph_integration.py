from __future__ import annotations

from ragless import Context, Retriever
from ragless.graph import LinkEdges, build_default_edge_builders


def test_search_graph_true_adds_linked_result_not_in_plain_top_k(linked_docs):
    r = Retriever(str(linked_docs))
    plain = r.search("password login", top_k=1, fetch_k=1)
    assert len(plain) == 1
    assert not plain[0].path.endswith("retry.md")

    expanded = r.search("password login", top_k=5, fetch_k=1, graph=True)
    paths = [h.path for h in expanded]
    assert any(p.endswith("retry.md") for p in paths)


def test_search_graph_results_carry_graph_reasons(linked_docs):
    r = Retriever(str(linked_docs))
    results = r.search("password login", top_k=5, fetch_k=1, graph=True)
    retry_hit = next(h for h in results if h.path.endswith("retry.md"))
    assert retry_hit.graph_reasons
    assert "auth.md" in retry_hit.graph_reasons[0]
    seed_hit = next(h for h in results if h.path.endswith("auth.md"))
    assert seed_hit.graph_reasons == []


def test_search_graph_false_never_sets_graph_reasons(sample_corpus):
    r = Retriever(str(sample_corpus))
    for h in r.search("installation", top_k=5):
        assert h.graph_reasons == []


def test_search_graph_true_respects_top_k(linked_docs):
    r = Retriever(str(linked_docs))
    results = r.search("password login", top_k=2, fetch_k=1, graph=True, max_expanded=20)
    assert len(results) <= 2


def test_search_graph_keep_seeds_true_never_drops_original_seed(linked_docs):
    r = Retriever(str(linked_docs))
    seed = r.search("password login", top_k=1, fetch_k=1)[0]
    results = r.search("password login", top_k=100, fetch_k=1, graph=True, keep_seeds=True)
    assert any(h.chunk.id == seed.chunk.id for h in results)


def test_search_graph_bfs_mode_works(linked_docs):
    r = Retriever(str(linked_docs))
    results = r.search("password login", top_k=5, fetch_k=1, graph=True, graph_mode="bfs")
    assert any(h.path.endswith("retry.md") for h in results)


def test_search_graph_edge_weights_can_disable_a_type(linked_docs):
    r = Retriever(str(linked_docs))
    with_links = r.search("password login", top_k=5, fetch_k=1, graph=True)
    without_links = r.search(
        "password login", top_k=5, fetch_k=1, graph=True, edge_weights={"link": 0.0}
    )
    assert any(h.path.endswith("retry.md") for h in with_links)
    assert not any(h.path.endswith("retry.md") for h in without_links)


def test_search_graph_rerank_combo_does_not_crash(fake_embedder, linked_docs):
    r = Retriever(str(linked_docs), engines=["bm25", "fuzzy", "semantic"], embedder=fake_embedder)
    results = r.search("password login", top_k=5, fetch_k=1, graph=True, rerank=True)
    assert results


def test_search_graph_true_on_in_memory_text():
    r = Retriever(
        [
            "Users authenticate with a password and session token.",
            "Session tokens expire after rotation and must be refreshed.",
            "Completely unrelated note about the weather today.",
        ]
    )
    results = r.search("password authentication", top_k=5, fetch_k=1, graph=True)
    assert results
    assert all(h.path is None for h in results)


# --- context(graph=True) ---------------------------------------------------------


def test_context_graph_true_pulls_in_linked_passage(linked_docs):
    r = Retriever(str(linked_docs))
    ctx = r.context("password login", max_tokens=2000, fetch_k=1, top_k=1, graph=True)
    assert isinstance(ctx, Context)
    paths = [p.path for p in ctx.passages]
    assert any(p and p.endswith("retry.md") for p in paths)
    retry_passage = next(p for p in ctx.passages if p.path and p.path.endswith("retry.md"))
    assert any("graph:" in reason for reason in retry_passage.reasons)


def test_context_graph_passages_still_get_window_merge_dedup(linked_docs):
    r = Retriever(str(linked_docs))
    ctx = r.context("password login", max_tokens=2000, fetch_k=1, top_k=1, graph=True, window=1)
    # every passage must still have valid, in-bounds line ranges after the
    # rest of the pipeline ran on top of the graph-expanded ones
    for p in ctx.passages:
        assert p.start_line >= 1
        assert p.end_line >= p.start_line


def test_context_graph_true_with_explicit_pipeline_has_no_automatic_effect(linked_docs):
    from ragless.context.pipeline import Pack

    r = Retriever(str(linked_docs))
    ctx = r.context(
        "password login", max_tokens=2000, fetch_k=1, top_k=1, graph=True, pipeline=[Pack()]
    )
    # graph=True is ignored because an explicit pipeline was given
    assert not any(p.path and p.path.endswith("retry.md") for p in ctx.passages)


def test_context_graph_true_custom_pipeline_can_add_graph_expand_itself(linked_docs):
    from ragless.context.pipeline import Pack
    from ragless.graph import GraphExpand

    r = Retriever(str(linked_docs))
    ctx = r.context(
        "password login",
        max_tokens=2000,
        fetch_k=1,
        top_k=1,
        pipeline=[GraphExpand(r.graph), Pack()],
    )
    assert any(p.path and p.path.endswith("retry.md") for p in ctx.passages)


def test_context_graph_false_no_graph_expand_step_runs(linked_docs):
    r = Retriever(str(linked_docs))
    ctx = r.context("password login", max_tokens=2000, fetch_k=1, top_k=1, graph=False)
    assert not any(p.path and p.path.endswith("retry.md") for p in ctx.passages)


# --- Retriever.graph property, edge_builders override ------------------------------


def test_retriever_graph_property_is_memoized(sample_corpus):
    r = Retriever(str(sample_corpus))
    g1 = r.graph
    g2 = r.graph
    assert g1 is g2


def test_retriever_edge_builders_override(linked_docs):
    r = Retriever(str(linked_docs), edge_builders=[LinkEdges(weight=0.1)])
    assert r.graph.edges
    assert all(e.weight <= 0.1 for e in r.graph.edges)
    assert all(e.type == "link" for e in r.graph.edges)


def test_retriever_graph_persists_to_cache_dir(linked_docs, tmp_path):
    cache = tmp_path / "cache"
    r = Retriever(str(linked_docs), cache_dir=str(cache))
    _ = r.graph
    assert (cache / "graph.json").exists()


def test_retriever_without_semantic_engine_has_no_semantic_knn_builder(sample_corpus):
    r = Retriever(str(sample_corpus))
    assert not any(type(b).__name__ == "SemanticKNNEdges" for b in r._edge_builders_override or [])
    assert not any(e.type == "semantic_knn" for e in r.graph.edges)


def test_retriever_with_semantic_engine_includes_semantic_knn_builder(fake_embedder, tmp_path):
    (tmp_path / "a.txt").write_text("pandas dataframe analysis tutorial content")
    (tmp_path / "b.txt").write_text("pandas dataframe analysis guide content")
    r = Retriever(str(tmp_path), engines=["bm25", "fuzzy", "semantic"], embedder=fake_embedder)
    builders = r._edge_builders_override or build_default_edge_builders(r._semantic_embedder())
    assert any(type(b).__name__ == "SemanticKNNEdges" for b in builders)

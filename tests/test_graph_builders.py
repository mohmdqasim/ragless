from __future__ import annotations

from ragless import Retriever
from ragless.graph.builders import (
    ImportEdges,
    LinkEdges,
    SemanticKNNEdges,
    SharedTermEdges,
    StructureEdges,
    SymbolEdges,
    make_build_context,
)


def _graph_for(path, builders):
    from ragless.graph.chunk_graph import ChunkGraph

    r = Retriever(str(path))
    return ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, builders), r


# --- Links -------------------------------------------------------------------


def test_link_edges_resolve_relative_markdown_links(linked_docs):
    graph, r = _graph_for(linked_docs, [LinkEdges()])
    reasons = [e.reason for e in graph.edges]
    assert any("auth.md" in r_ and "retry.md" in r_ and "links to" in r_ for r_ in reasons)


def test_link_edges_resolve_to_heading_anchor_chunk(linked_docs):
    graph, r = _graph_for(linked_docs, [LinkEdges()])
    auth_chunk = next(c for c in r._chunks if c.path.endswith("auth.md"))
    retry_backoff_chunk = next(
        c for c in r._chunks if c.path.endswith("retry.md") and "Backoff" in c.text
    )
    edge = next(e for e in graph.edges if e.src == auth_chunk.id)
    assert edge.dst == retry_backoff_chunk.id


def test_link_edges_resolve_bare_relative_path_mention(linked_docs):
    graph, _ = _graph_for(linked_docs, [LinkEdges()])
    reasons = [e.reason for e in graph.edges]
    assert any("support.md" in r_ and "policy.md" in r_ for r_ in reasons)


def test_link_edges_skip_broken_links_without_crashing(tmp_path):
    (tmp_path / "a.md").write_text("See [missing](./nope.md) for more.\n")
    graph, _ = _graph_for(tmp_path, [LinkEdges()])
    assert graph.edges == []
    targets = next(iter(graph.unresolved.get("link", {}).values()), set())
    assert any("nope.md" in t for t in targets)


def test_link_edges_ignore_external_urls(tmp_path):
    (tmp_path / "a.md").write_text("See [docs](https://example.com/page) for more.\n")
    (tmp_path / "b.md").write_text("Some other content.\n")
    graph, _ = _graph_for(tmp_path, [LinkEdges()])
    assert graph.edges == []


def test_link_edges_no_self_edges(tmp_path):
    (tmp_path / "a.md").write_text("Self reference: [here](./a.md).\n")
    graph, _ = _graph_for(tmp_path, [LinkEdges()])
    assert graph.edges == []


# --- Imports -------------------------------------------------------------------


def test_import_edges_resolve_python_from_import(python_project):
    graph, r = _graph_for(python_project, [ImportEdges()])
    reasons = [e.reason for e in graph.edges]
    assert any("client.py" in r_ and "util.py" in r_ and "imports" in r_ for r_ in reasons)


def test_import_edges_resolve_relative_python_import(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("X = 1\n")
    (tmp_path / "pkg" / "b.py").write_text("from . import a\n")
    graph, _ = _graph_for(tmp_path, [ImportEdges()])
    reasons = [e.reason for e in graph.edges]
    assert any("b.py" in r_ and "a.py" in r_ for r_ in reasons)


def test_import_edges_skip_unresolvable_bare_imports(tmp_path):
    (tmp_path / "a.py").write_text("import numpy\nimport os\n")
    graph, _ = _graph_for(tmp_path, [ImportEdges()])
    assert graph.edges == []


def test_import_edges_never_crash_on_syntax_error(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n  pass\nimport os\n")
    (tmp_path / "ok.py").write_text("import broken\n")
    graph, r = _graph_for(tmp_path, [ImportEdges()])
    broken_chunk = next(c for c in r._chunks if c.path.endswith("broken.py"))
    # scanning broken.py itself for its *own* imports is safely skipped (no crash,
    # no edges *from* it), but another file importing *into* broken.py still
    # resolves fine since that's pure path resolution, not a parse of its content.
    assert all(e.src != broken_chunk.id for e in graph.edges)
    assert any(e.dst == broken_chunk.id for e in graph.edges)


def test_import_edges_resolve_js_relative_require(tmp_path):
    (tmp_path / "util.js").write_text("module.exports.add = (a, b) => a + b;\n")
    (tmp_path / "main.js").write_text("const { add } = require('./util');\nadd(1, 2);\n")
    graph, _ = _graph_for(tmp_path, [ImportEdges()])
    reasons = [e.reason for e in graph.edges]
    assert any("main.js" in r_ and "util.js" in r_ for r_ in reasons)


def test_import_edges_skip_bare_js_package_specifiers(tmp_path):
    (tmp_path / "main.js").write_text("import React from 'react';\n")
    graph, _ = _graph_for(tmp_path, [ImportEdges()])
    assert graph.edges == []
    assert graph.unresolved == {}


# --- Symbols -----------------------------------------------------------------


def test_symbol_edges_connect_python_def_to_usage(python_project):
    graph, r = _graph_for(python_project, [SymbolEdges(min_name_length=3)])
    util_def_chunk = next(
        c for c in r._chunks if c.path.endswith("util.py") and "def backoff" in c.text
    )
    client_chunk = next(c for c in r._chunks if c.path.endswith("client.py"))
    assert any(e.src == util_def_chunk.id and e.dst == client_chunk.id for e in graph.edges)


def test_symbol_edges_skip_common_short_names(tmp_path):
    (tmp_path / "a.py").write_text("def run():\n    pass\n")
    (tmp_path / "b.py").write_text("run()\n")
    graph, _ = _graph_for(tmp_path, [SymbolEdges()])
    assert graph.edges == []  # "run" is in the common-name stoplist


def test_symbol_edges_cap_usage_edges_per_symbol(tmp_path):
    (tmp_path / "defs.py").write_text("def distinctive_helper():\n    pass\n")
    for i in range(40):
        (tmp_path / f"usage_{i}.py").write_text("distinctive_helper()\n")
    graph, _ = _graph_for(tmp_path, [SymbolEdges(max_usage_edges=5, max_usage_fraction=1.0)])
    assert len(graph.edges) == 5


def test_symbol_edges_regex_definitions_for_js(tmp_path):
    (tmp_path / "a.js").write_text("function distinctiveName() {\n  return 1;\n}\n")
    (tmp_path / "b.js").write_text("distinctiveName();\n")
    graph, _ = _graph_for(tmp_path, [SymbolEdges()])
    assert any(e.reason == "defines 'distinctiveName', used here" for e in graph.edges)


# --- Structure -----------------------------------------------------------------


def test_structure_edges_sequential_neighbors(tmp_path):
    text = "\n".join(f"line {i}" for i in range(1, 20))
    (tmp_path / "a.md").write_text(text)
    from ragless.chunking import LineChunker

    r = Retriever(str(tmp_path), chunker=LineChunker(lines_per_chunk=5, overlap=1))
    from ragless.graph.chunk_graph import ChunkGraph

    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, [StructureEdges()])
    assert any(e.type == "structure" and "sequential neighbor" in e.reason for e in graph.edges)


def test_structure_edges_heading_hierarchy(linked_docs):
    from ragless.chunking import LineChunker
    from ragless.graph.chunk_graph import ChunkGraph

    # a tiny chunker so "Retry policy" and "Backoff" land in different chunks
    r = Retriever(str(linked_docs), chunker=LineChunker(lines_per_chunk=2, overlap=0))
    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, [StructureEdges()])
    reasons = [e.reason for e in graph.edges if "heading hierarchy" in e.reason]
    assert any("Retry policy" in r_ and "Backoff" in r_ for r_ in reasons)


# --- Shared rare terms ------------------------------------------------------------


def test_shared_term_edges_connect_distinctive_overlap(tmp_path):
    (tmp_path / "a.txt").write_text("the quokka wombat marsupial lives in australia")
    (tmp_path / "b.txt").write_text("the quokka wombat marsupial is a cute animal")
    (tmp_path / "c.txt").write_text("completely different unrelated boring filler sentence")
    graph, _ = _graph_for(tmp_path, [SharedTermEdges(min_overlap=2)])
    assert len(graph.edges) >= 1
    assert all(e.type == "shared_term" for e in graph.edges)


def test_shared_term_edges_respect_max_edges_per_node(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("quokka wombat marsupial zyzzyva distinctive rareterm")
    graph, _ = _graph_for(tmp_path, [SharedTermEdges(max_edges_per_node=2, min_overlap=1)])
    from collections import Counter

    degree = Counter()
    for e in graph.edges:
        degree[e.src] += 1
        degree[e.dst] += 1
    assert all(d <= 2 for d in degree.values())


def test_shared_term_edges_no_crash_on_tiny_corpus(tmp_path):
    (tmp_path / "a.txt").write_text("only one file")
    graph, _ = _graph_for(tmp_path, [SharedTermEdges()])
    assert graph.edges == []


# --- Semantic kNN -----------------------------------------------------------------


def test_semantic_knn_edges_connect_similar_chunks(fake_embedder, tmp_path):
    (tmp_path / "a.txt").write_text("pandas dataframe analysis tutorial")
    (tmp_path / "b.txt").write_text("pandas dataframe analysis guide")
    (tmp_path / "c.txt").write_text("completely unrelated rocket engine telemetry")
    graph, _ = _graph_for(tmp_path, [SemanticKNNEdges(fake_embedder, min_similarity=0.5, k=2)])
    assert any(e.type == "semantic_knn" for e in graph.edges)


def test_semantic_knn_edges_no_duplicate_pairs(fake_embedder, tmp_path):
    (tmp_path / "a.txt").write_text("pandas dataframe analysis tutorial")
    (tmp_path / "b.txt").write_text("pandas dataframe analysis guide")
    graph, _ = _graph_for(tmp_path, [SemanticKNNEdges(fake_embedder, min_similarity=0.0, k=5)])
    pairs = [frozenset((e.src, e.dst)) for e in graph.edges]
    assert len(pairs) == len(set(pairs))


# --- BuildContext / make_build_context ---------------------------------------------


def test_make_build_context_groups_chunks_by_doc_in_order(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = make_build_context(r._chunks, r._chunk_doc_id, r._doc_text)
    for _doc_id, chunks in ctx.chunks_by_doc.items():
        lines = [c.start_line for c in chunks]
        assert lines == sorted(lines)


def test_make_build_context_only_source_paths_filters_sources(sample_corpus):
    r = Retriever(str(sample_corpus))
    only = {next(iter(r._doc_text))}
    ctx = make_build_context(r._chunks, r._chunk_doc_id, r._doc_text, only_source_paths=only)
    assert ctx.sources() == list(only)

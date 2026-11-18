"""End-to-end regression test for the SemanticKNNEdges re-embedding bug.

Covers the exact reported scenario: ``engines=["bm25", "fuzzy", "semantic"]``
with a warm ``cache_dir`` -- the ``semantic`` engine's own index load must
not be duplicated by ``r.graph``, a warm restart must embed nothing, a cold
start must embed each chunk exactly once (not twice), and an edit to one
file must only re-embed that file's chunks on the next restart.
"""

from __future__ import annotations

import time

from ragless import Retriever


def _corpus(tmp_path, n=20):
    for i in range(n):
        (tmp_path / f"doc_{i}.md").write_text(
            f"document {i} discusses retry policy and backoff schedule number {i}"
        )


def test_cold_start_embeds_each_chunk_exactly_once(counting_embedder, tmp_path):
    _corpus(tmp_path)
    r = Retriever(str(tmp_path), engines=["bm25", "fuzzy", "semantic"], embedder=counting_embedder)
    _ = r.graph  # triggers SemanticKNNEdges

    expected = sorted(c.text for c in r._chunks)
    embedded = sorted(counting_embedder.embedded_texts)
    assert embedded == expected  # exactly once each, not twice


def test_warm_restart_embeds_nothing(counting_embedder, tmp_path):
    cache_dir = tmp_path / ".cache"
    _corpus(tmp_path)

    r1 = Retriever(
        str(tmp_path),
        engines=["bm25", "fuzzy", "semantic"],
        embedder=counting_embedder,
        cache_dir=str(cache_dir),
    )
    _ = r1.graph
    assert counting_embedder.embedded_texts  # sanity: the cold run did embed something
    counting_embedder.reset_calls()

    r2 = Retriever(
        str(tmp_path),
        engines=["bm25", "fuzzy", "semantic"],
        embedder=counting_embedder,
        cache_dir=str(cache_dir),
    )
    _ = r2.graph

    assert counting_embedder.embedded_texts == []
    assert len(r2.graph.edges) == len(r1.graph.edges)  # same graph, just reconstructed warm


def test_editing_one_file_only_reembeds_that_files_chunks(counting_embedder, tmp_path):
    cache_dir = tmp_path / ".cache"
    _corpus(tmp_path)

    r1 = Retriever(
        str(tmp_path),
        engines=["bm25", "fuzzy", "semantic"],
        embedder=counting_embedder,
        cache_dir=str(cache_dir),
    )
    _ = r1.graph
    counting_embedder.reset_calls()

    time.sleep(0.01)
    edited_path = tmp_path / "doc_5.md"
    edited_path.write_text("document 5 now talks about something totally different")

    r2 = Retriever(
        str(tmp_path),
        engines=["bm25", "fuzzy", "semantic"],
        embedder=counting_embedder,
        cache_dir=str(cache_dir),
    )
    _ = r2.graph

    edited_chunk_texts = {c.text for c in r2._chunks if c.path == str(edited_path)}
    assert edited_chunk_texts  # the file still produced at least one chunk
    assert set(counting_embedder.embedded_texts) == edited_chunk_texts


def test_vectors_by_chunk_id_matches_search_results(fake_embedder, tmp_path):
    _corpus(tmp_path, n=5)
    r = Retriever(str(tmp_path), engines=["semantic"], embedder=fake_embedder)
    vectors = r._retrievers["semantic"].vectors_by_chunk_id()
    assert set(vectors) == {c.id for c in r._chunks}
    for vec in vectors.values():
        norm = float((vec**2).sum() ** 0.5)
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0  # already unit-normalized

from __future__ import annotations

import os
import time

from ragless import Retriever
from ragless.graph import ChunkGraph, build_default_edge_builders
from ragless.graph.builders import SharedTermEdges, SymbolEdges


def _edgeset(graph):
    return sorted((e.src, e.dst, e.type, round(e.weight, 6), e.reason) for e in graph.edges)


def _full_rebuild(path, builders=None):
    r = Retriever(str(path))
    return ChunkGraph.build(
        r._chunks, r._chunk_doc_id, r._doc_text, builders or build_default_edge_builders()
    )


def test_incremental_matches_full_rebuild_after_content_edit(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nSee [b](./b.md) for details.\n")
    (docs / "b.md").write_text("# B\n\noriginal content\n")
    (docs / "c.md").write_text("# C\n\nunrelated\n")

    _ = Retriever(str(docs), cache_dir=str(cache)).graph  # populate cache
    time.sleep(0.01)
    (docs / "b.md").write_text("# B\n\nupdated content, now mentions [c](./c.md) too.\n")

    incremental = Retriever(str(docs), cache_dir=str(cache)).graph
    assert _edgeset(incremental) == _edgeset(_full_rebuild(docs))


def test_incremental_resolves_previously_broken_link_after_add(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("See [missing](./missing.md) and [b](./b.md).\n")
    (docs / "b.md").write_text("content b\n")

    g1 = Retriever(str(docs), cache_dir=str(cache)).graph
    assert len(g1.edges) == 1  # only a->b resolves so far

    time.sleep(0.01)
    (docs / "missing.md").write_text("now it exists\n")
    incremental = Retriever(str(docs), cache_dir=str(cache)).graph
    assert _edgeset(incremental) == _edgeset(_full_rebuild(docs))
    assert len(incremental.edges) == 2


def test_incremental_matches_full_rebuild_after_delete(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("See [b](./b.md).\n")
    (docs / "b.md").write_text("content b, see [c](./c.md).\n")
    (docs / "c.md").write_text("content c\n")

    _ = Retriever(str(docs), cache_dir=str(cache)).graph
    time.sleep(0.01)
    os.remove(docs / "c.md")
    (docs / "b.md").write_text("content b, c is gone now.\n")

    incremental = Retriever(str(docs), cache_dir=str(cache)).graph
    assert _edgeset(incremental) == _edgeset(_full_rebuild(docs))


def test_incremental_matches_full_rebuild_after_rename(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("See [b](./old_name.md).\n")
    (docs / "old_name.md").write_text("content\n")

    _ = Retriever(str(docs), cache_dir=str(cache)).graph
    time.sleep(0.01)
    os.rename(docs / "old_name.md", docs / "new_name.md")
    (docs / "a.md").write_text("See [b](./new_name.md).\n")

    incremental = Retriever(str(docs), cache_dir=str(cache)).graph
    assert _edgeset(incremental) == _edgeset(_full_rebuild(docs))
    assert len(incremental.edges) == 1


def test_incremental_noop_reuses_cache_unchanged(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("See [b](./b.md).\n")
    (docs / "b.md").write_text("content\n")

    first = Retriever(str(docs), cache_dir=str(cache)).graph
    second = Retriever(str(docs), cache_dir=str(cache)).graph
    assert _edgeset(first) == _edgeset(second) == _edgeset(_full_rebuild(docs))


def test_incremental_global_builders_stay_correct_after_edit(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("quokka wombat marsupial distinctive rareterm content")
    (docs / "b.txt").write_text("unrelated filler text about nothing special")

    builders = [SharedTermEdges(min_overlap=1)]
    r1 = Retriever(str(docs), cache_dir=str(cache))
    g1 = ChunkGraph.build(r1._chunks, r1._chunk_doc_id, r1._doc_text, builders)
    g1.save(cache)

    time.sleep(0.01)
    (docs / "b.txt").write_text("now also mentions quokka wombat marsupial distinctive rareterm")

    r2 = Retriever(str(docs), cache_dir=str(cache))
    cached = ChunkGraph.load(cache)
    updated = cached.update(
        r2._chunks, r2._chunk_doc_id, r2._doc_text, builders, r2._file_signatures
    )

    full = ChunkGraph.build(r2._chunks, r2._chunk_doc_id, r2._doc_text, builders)
    assert _edgeset(updated) == _edgeset(full)
    assert len(updated.edges) > 0  # the new shared-term overlap was actually picked up


def test_incremental_symbol_builder_stays_correct_after_edit(tmp_path):
    (tmp_path / "defs.py").write_text("def distinctive_symbol():\n    pass\n")
    (tmp_path / "usage.py").write_text("print('nothing yet')\n")

    builders = [SymbolEdges()]
    r1 = Retriever(str(tmp_path))
    g1 = ChunkGraph.build(r1._chunks, r1._chunk_doc_id, r1._doc_text, builders)
    assert g1.edges == []

    (tmp_path / "usage.py").write_text("distinctive_symbol()\n")
    r2 = Retriever(str(tmp_path))
    updated = g1.update(r2._chunks, r2._chunk_doc_id, r2._doc_text, builders, {})
    full = ChunkGraph.build(r2._chunks, r2._chunk_doc_id, r2._doc_text, builders)
    assert _edgeset(updated) == _edgeset(full)
    assert len(updated.edges) == 1


def test_incremental_in_memory_text_always_rescanned(tmp_path):
    cache = tmp_path / ".cache"
    from ragless.graph.builders import LinkEdges

    builders = [LinkEdges()]
    r1 = Retriever(["first version, no links here"])
    g1 = ChunkGraph.build(r1._chunks, r1._chunk_doc_id, r1._doc_text, builders)
    g1.save(cache)

    # completely different in-memory content next run; mem: has no signature to
    # diff, so it must always be rescanned rather than trusting the stale cache.
    r2 = Retriever(["second version, totally different content"])
    cached = ChunkGraph.load(cache)
    updated = cached.update(r2._chunks, r2._chunk_doc_id, r2._doc_text, builders, {})
    assert updated.chunk_doc_id == r2._chunk_doc_id


def test_builder_fingerprint_mismatch_forces_full_rebuild(tmp_path):
    from ragless.graph import build_or_update_graph
    from ragless.graph.builders import LinkEdges

    cache = tmp_path / ".cache"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("See [b](./b.md).\n")
    (tmp_path / "docs" / "b.md").write_text("content\n")
    docs = tmp_path / "docs"

    r1 = Retriever(str(docs))
    build_or_update_graph(
        cache, r1._chunks, r1._chunk_doc_id, r1._doc_text, [LinkEdges()], r1._file_signatures
    )

    r2 = Retriever(str(docs))
    # different builder config (e.g. a different weight) -> fingerprint differs
    graph = build_or_update_graph(
        cache,
        r2._chunks,
        r2._chunk_doc_id,
        r2._doc_text,
        [LinkEdges(weight=0.5)],
        r2._file_signatures,
    )
    assert graph.edges and graph.edges[0].weight == 0.5


# --- structural edge cases: cycles, self-links, huge hubs, binary/unicode --------


def test_incremental_handles_cyclic_links(tmp_path):
    cache = tmp_path / ".cache"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("See [b](./b.md).\n")
    (docs / "b.md").write_text("See [a](./a.md).\n")

    _ = Retriever(str(docs), cache_dir=str(cache)).graph
    time.sleep(0.01)
    (docs / "a.md").write_text("See [b](./b.md) again, edited.\n")
    incremental = Retriever(str(docs), cache_dir=str(cache)).graph
    assert _edgeset(incremental) == _edgeset(_full_rebuild(docs))


def test_graph_build_ignores_binary_files(tmp_path):
    (tmp_path / "readable.md").write_text("See [other](./other.md).\n")
    (tmp_path / "other.md").write_text("content\n")
    (tmp_path / "image.bin").write_bytes(bytes([0, 1, 2, 0, 255, 254, 0, 0]))
    r = Retriever(str(tmp_path))
    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, build_default_edge_builders())
    assert all(
        not e.src.endswith("image.bin") and not e.dst.endswith("image.bin") for e in graph.edges
    )


def test_graph_build_handles_unicode_content(tmp_path):
    (tmp_path / "a.md").write_text("café naïve 你好 see [b](./b.md)\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("résumé content\n", encoding="utf-8")
    r = Retriever(str(tmp_path))
    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, build_default_edge_builders())
    assert any(e.type == "link" for e in graph.edges)


def test_graph_build_no_edges_at_all_does_not_crash(tmp_path):
    (tmp_path / "a.txt").write_text("nothing links to anything")
    r = Retriever(str(tmp_path))
    graph = ChunkGraph.build(r._chunks, r._chunk_doc_id, r._doc_text, build_default_edge_builders())
    assert graph.edges == []
    assert graph.stats()["edges"] == 0

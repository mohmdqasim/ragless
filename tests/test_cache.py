from __future__ import annotations

import numpy as np

from ragless.cache import EmbeddingCache, file_signature
from ragless.types import Chunk


class _StubEmbedder:
    def __init__(self, name="stub", dim=4):
        self.name = name
        self.dim = dim


def _chunks(path):
    return [
        Chunk(id="c0", text="hello", path=path, start_line=1, end_line=1),
        Chunk(id="c1", text="world", path=path, start_line=2, end_line=2),
    ]


def test_put_then_get_roundtrips(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("hello\nworld\n")
    cache = EmbeddingCache(tmp_path / "cache")
    embedder = _StubEmbedder()
    chunks = _chunks(str(source))
    vectors = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    sig = file_signature(source)
    cache.put(str(source), sig, embedder, chunks, vectors)
    loaded = cache.get(str(source), sig, embedder, chunks)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, vectors)


def test_get_misses_when_nothing_cached(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    embedder = _StubEmbedder()
    source = tmp_path / "doc.txt"
    source.write_text("hello\n")
    chunks = _chunks(str(source))
    assert cache.get(str(source), file_signature(source), embedder, chunks) is None


def test_invalidates_on_changed_file_signature(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("hello\nworld\n")
    cache = EmbeddingCache(tmp_path / "cache")
    embedder = _StubEmbedder()
    chunks = _chunks(str(source))
    vectors = np.zeros((2, 4), dtype=np.float32)

    sig = file_signature(source)
    cache.put(str(source), sig, embedder, chunks, vectors)

    # Simulate the file changing: a different (mtime, size) signature.
    new_sig = (sig[0] + 100, sig[1] + 1)
    assert cache.get(str(source), new_sig, embedder, chunks) is None


def test_invalidates_on_changed_chunk_ids(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("hello\nworld\n")
    cache = EmbeddingCache(tmp_path / "cache")
    embedder = _StubEmbedder()
    sig = file_signature(source)
    chunks = _chunks(str(source))
    vectors = np.zeros((2, 4), dtype=np.float32)
    cache.put(str(source), sig, embedder, chunks, vectors)

    different_chunks = [
        Chunk(id="different", text="hello", path=str(source), start_line=1, end_line=1),
        Chunk(id="c1", text="world", path=str(source), start_line=2, end_line=2),
    ]
    assert cache.get(str(source), sig, embedder, different_chunks) is None


def test_invalidates_on_changed_embedder(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("hello\nworld\n")
    cache = EmbeddingCache(tmp_path / "cache")
    sig = file_signature(source)
    chunks = _chunks(str(source))
    vectors = np.zeros((2, 4), dtype=np.float32)
    cache.put(str(source), sig, _StubEmbedder(name="model-a"), chunks, vectors)

    assert cache.get(str(source), sig, _StubEmbedder(name="model-b"), chunks) is None
    assert cache.get(str(source), sig, _StubEmbedder(name="model-a", dim=8), chunks) is None


def test_different_files_do_not_collide(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello\n")
    b.write_text("hello\n")
    cache = EmbeddingCache(tmp_path / "cache")
    embedder = _StubEmbedder()

    chunks_a = [Chunk(id="c0", text="hello", path=str(a), start_line=1, end_line=1)]
    chunks_b = [Chunk(id="c0", text="hello", path=str(b), start_line=1, end_line=1)]
    vec_a = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    vec_b = np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32)

    cache.put(str(a), file_signature(a), embedder, chunks_a, vec_a)
    cache.put(str(b), file_signature(b), embedder, chunks_b, vec_b)

    np.testing.assert_array_equal(cache.get(str(a), file_signature(a), embedder, chunks_a), vec_a)
    np.testing.assert_array_equal(cache.get(str(b), file_signature(b), embedder, chunks_b), vec_b)

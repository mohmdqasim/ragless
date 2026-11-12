from __future__ import annotations

import numpy as np

from ragless.cache import EmbeddingCache, file_signature
from ragless.embedders.base import Embedder
from ragless.retrievers.semantic import SemanticRetriever
from ragless.types import Chunk


class _CountingEmbedder(Embedder):
    """Deterministic bag-of-words embedder that counts embed() calls."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.name = f"counting:{dim}"
        self.calls = 0
        self._vocab: dict[str, int] = {}

    def _index_for(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) % self.dim
        return self._vocab[word]

    def embed(self, texts):
        self.calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, self._index_for(word)] += 1.0
        return vectors


def test_ranks_by_cosine_similarity(fake_embedder):
    chunks = [
        Chunk(id="c0", text="pandas dataframe analysis", path=None, start_line=1, end_line=1),
        Chunk(
            id="c1",
            text="completely different unrelated topic",
            path=None,
            start_line=1,
            end_line=1,
        ),
    ]
    retriever = SemanticRetriever(fake_embedder)
    retriever.index(chunks)

    results = retriever.query("pandas dataframe", top_k=2)
    assert results[0][0] == "c0"


def test_empty_chunks():
    retriever = SemanticRetriever(_CountingEmbedder())
    retriever.index([])
    assert retriever.query("anything", top_k=5) == []


def test_empty_query(fake_embedder):
    retriever = SemanticRetriever(fake_embedder)
    retriever.index([Chunk(id="c0", text="hello", path=None, start_line=1, end_line=1)])
    assert retriever.query("", top_k=5) == []


def test_cache_reuses_unchanged_file_embeddings(tmp_path):
    embedder = _CountingEmbedder()
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(cache_dir)

    unchanged = tmp_path / "unchanged.txt"
    unchanged.write_text("alpha beta\n")
    changed = tmp_path / "changed.txt"
    changed.write_text("gamma delta\n")

    def build_chunks():
        return [
            Chunk(id="u0", text="alpha beta", path=str(unchanged), start_line=1, end_line=1),
            Chunk(id="c0", text="gamma delta", path=str(changed), start_line=1, end_line=1),
        ]

    signatures = {
        str(unchanged): file_signature(unchanged),
        str(changed): file_signature(changed),
    }

    retriever = SemanticRetriever(embedder, cache=cache)
    retriever.index_with_signatures(build_chunks(), signatures)
    assert embedder.calls == 2  # one embed() call per distinct source file

    # Modify only "changed.txt" and rebuild the signature for it.
    changed.write_text("gamma delta epsilon\n")
    new_chunks = [
        Chunk(id="u0", text="alpha beta", path=str(unchanged), start_line=1, end_line=1),
        Chunk(id="c0", text="gamma delta epsilon", path=str(changed), start_line=1, end_line=1),
    ]
    new_signatures = {
        str(unchanged): file_signature(unchanged),
        str(changed): file_signature(changed),
    }

    embedder.calls = 0
    retriever.index_with_signatures(new_chunks, new_signatures)
    # Only the changed file should have triggered a fresh embed() call.
    assert embedder.calls == 1


def test_index_with_signatures_handles_in_memory_chunks(fake_embedder):
    cache = None
    retriever = SemanticRetriever(fake_embedder, cache=cache)
    chunks = [Chunk(id="m0", text="in memory text", path=None, start_line=1, end_line=1)]
    retriever.index_with_signatures(chunks, {})
    results = retriever.query("in memory text", top_k=1)
    assert results[0][0] == "m0"

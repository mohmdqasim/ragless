from __future__ import annotations

import numpy as np
import pytest

from ragless.embedders.base import Embedder


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that download and run a real embedding model.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow (downloads a real model)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


class FakeEmbedder(Embedder):
    """Deterministic, dependency-free stand-in for a real embedding model.

    Produces a bag-of-words vector (one dimension per distinct word, assigned
    on first sight) so that semantically similar text (sharing words) lands
    close together under cosine similarity, without needing to download a
    real model in tests. Unlike hashing words with the builtin ``hash()``,
    this doesn't depend on Python's per-process hash randomization, so it's
    reproducible across runs/processes/machines.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"fake:{dim}"
        self._vocab: dict[str, int] = {}

    def _index_for(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) % self.dim
        return self._vocab[word]

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, self._index_for(word)] += 1.0
        return vectors


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_corpus(tmp_path):
    """A small on-disk corpus with predictable, searchable content."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "install.md").write_text(
        "# Installation\n"
        "\n"
        "Run pip install ragless to get started.\n"
        "Then import the Retriever class and point it at a folder.\n"
    )
    (docs / "search.md").write_text(
        "# Searching\n"
        "\n"
        "Use the search method with a query string.\n"
        "Hybrid mode combines bm25, fuzzy, and semantic retrieval.\n"
    )
    nested = docs / "guides"
    nested.mkdir()
    (nested / "caching.md").write_text(
        "# Caching\n"
        "\n"
        "Embeddings are cached as npy and json sidecar files.\n"
        "Only changed files are re-embedded on the next run.\n"
    )
    return docs

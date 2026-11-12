"""Semantic (embedding) retriever: NumPy cosine similarity, no vector DB."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragless.cache import EmbeddingCache, FileSignature
from ragless.embedders.base import Embedder
from ragless.retrievers.base import BaseRetriever
from ragless.types import Chunk


class SemanticRetriever(BaseRetriever):
    """Ranks chunks by cosine similarity between dense embeddings.

    Embeddings are plain NumPy arrays held in memory for search; there is no
    vector database involved. When an :class:`~ragless.cache.EmbeddingCache`
    is supplied, embeddings for unchanged files are loaded from disk instead
    of being recomputed.
    """

    name = "semantic"

    def __init__(self, embedder: Embedder, cache: EmbeddingCache | None = None) -> None:
        self.embedder = embedder
        self.cache = cache
        self._chunk_ids: list[str] = []
        self._vectors = np.zeros((0, embedder.dim), dtype=np.float32)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index ``chunks`` without per-file cache invalidation signatures.

        Used when a retriever is built standalone. :meth:`ragless.Retriever`
        instead calls :meth:`index_with_signatures` so unchanged files reuse
        cached embeddings.
        """
        self.index_with_signatures(chunks, {})

    def index_with_signatures(
        self,
        chunks: Sequence[Chunk],
        signatures: dict[str, FileSignature],
    ) -> None:
        if not chunks:
            self._chunk_ids = []
            self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
            return

        groups: dict[str | None, list[Chunk]] = {}
        order: list[str | None] = []
        for c in chunks:
            if c.path not in groups:
                groups[c.path] = []
                order.append(c.path)
            groups[c.path].append(c)

        all_ids: list[str] = []
        all_vectors: list[np.ndarray] = []
        for path in order:
            group_chunks = groups[path]
            signature = signatures.get(path) if path is not None else None
            vectors: np.ndarray | None = None
            if self.cache is not None and path is not None and signature is not None:
                vectors = self.cache.get(path, signature, self.embedder, group_chunks)
            if vectors is None:
                vectors = self.embedder.embed([c.text for c in group_chunks])
                if self.cache is not None and path is not None and signature is not None:
                    self.cache.put(path, signature, self.embedder, group_chunks, vectors)
            all_ids.extend(c.id for c in group_chunks)
            all_vectors.append(vectors)

        matrix = np.concatenate(all_vectors, axis=0).astype(np.float32)
        self._vectors = _normalize_rows(matrix)
        self._chunk_ids = all_ids

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self._chunk_ids or not query.strip():
            return []
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        norm = float(np.linalg.norm(q_vec))
        if norm == 0.0:
            return []
        q_vec = q_vec / norm

        scores = self._vectors @ q_vec
        k = min(top_k, scores.shape[0])
        if k < scores.shape[0]:
            idx = np.argpartition(-scores, k - 1)[:k]
        else:
            idx = np.arange(scores.shape[0])
        idx = idx[np.argsort(-scores[idx])]
        return [(self._chunk_ids[i], float(scores[i])) for i in idx]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms

"""Optional cross-encoder reranking of fused search results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragless.exceptions import MissingDependencyError
from ragless.types import Chunk

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Interface a reranker must satisfy to plug into :class:`ragless.Retriever`."""

    def rerank(
        self, query: str, chunks: Sequence[Chunk], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` pairs for ``chunks``, best first."""
        ...


class CrossEncoderReranker:
    """Reranks candidates with a small local cross-encoder via fastembed.

    Unlike the bi-encoder used for semantic search, a cross-encoder scores
    the query and each candidate jointly, which is slower but typically more
    accurate -- it is meant to be applied only to the top handful of fused
    candidates, not the whole corpus. Requires ``pip install ragless[semantic]``.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised without extra installed
            raise MissingDependencyError("Cross-encoder reranking", "semantic") from exc

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, chunks: Sequence[Chunk], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        scores = [float(s) for s in self._model.rerank(query, texts)]
        pairs = list(zip((c.id for c in chunks), scores, strict=True))
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:top_k] if top_k is not None else pairs

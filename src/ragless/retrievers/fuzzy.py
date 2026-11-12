"""Fuzzy string-similarity retriever backed by rapidfuzz."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rapidfuzz import fuzz, process

from ragless.retrievers.base import BaseRetriever
from ragless.types import Chunk


class FuzzyRetriever(BaseRetriever):
    """Ranks chunks by fuzzy string similarity to the query.

    Good at catching typos, partial matches, and near-duplicate phrasing that
    exact keyword search (BM25) would miss.
    """

    name = "fuzzy"

    def __init__(self, scorer: Callable[..., float] = fuzz.WRatio) -> None:
        self.scorer = scorer
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunk_ids = [c.id for c in chunks]
        self._texts = [c.text for c in chunks]

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self._texts or not query.strip():
            return []
        matches = process.extract(query, self._texts, scorer=self.scorer, limit=top_k)
        return [(self._chunk_ids[index], float(score)) for _, score, index in matches if score > 0]

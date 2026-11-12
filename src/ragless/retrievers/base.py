"""Base interface implemented by every retrieval backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragless.types import Chunk


class BaseRetriever(ABC):
    """A pluggable retrieval strategy.

    A retriever is given the full set of chunks once (:meth:`index`) and can
    then be queried repeatedly (:meth:`query`). Implementations should not
    mutate chunks; they only need to produce relevance rankings.
    """

    name: str

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the retriever's internal index over ``chunks``."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, score)`` pairs, best first.

        Scores are retriever-specific and not necessarily comparable across
        retrievers; fusion (see :mod:`ragless.fusion`) only relies on rank
        order, not on the raw magnitudes.
        """
        raise NotImplementedError

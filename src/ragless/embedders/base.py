"""Base interface implemented by every embedding backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    """A pluggable text-embedding backend.

    ``name`` must uniquely identify the model (and any settings that affect
    its output vectors) since it is part of the embedding cache key: changing
    models invalidates cached embeddings rather than silently reusing stale
    ones.
    """

    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into a ``(len(texts), self.dim)`` float32 array."""
        raise NotImplementedError

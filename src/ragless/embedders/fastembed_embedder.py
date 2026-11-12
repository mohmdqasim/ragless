"""Default semantic embedder: a small local model served via fastembed (ONNX runtime).

Requires the ``semantic`` extra: ``pip install ragless[semantic]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragless.embedders.base import Embedder
from ragless.exceptions import MissingDependencyError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder(Embedder):
    """Wraps :mod:`fastembed` to produce dense vectors with a small local model.

    No GPU or external service is required; models run locally via ONNX
    runtime and are cached on disk by fastembed after the first download.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised without extra installed
            raise MissingDependencyError("Semantic search", "semantic") from exc

        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = int(np.asarray(probe).shape[0])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._model.embed(list(texts)))
        return np.asarray(vectors, dtype=np.float32)

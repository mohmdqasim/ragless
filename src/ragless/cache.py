"""On-disk embedding cache: plain sidecar files, no database.

Each cached source file gets two sidecar files inside the cache directory:
a ``.npy`` holding the embedding matrix and a ``.json`` holding the metadata
needed to validate the cache (which embedder produced it, the source file's
mtime/size, and the exact chunk ids it covers). There is no cache index or
manifest to keep in sync -- each file's validity is self-contained, and a
changed source file simply gets new sidecar files on its next embed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ragless.types import Chunk

if TYPE_CHECKING:
    from ragless.embedders.base import Embedder

FileSignature = tuple[float, int]


def file_signature(path: Path) -> FileSignature:
    """A cheap (mtime, size) signature used to detect that a file changed."""
    stat = path.stat()
    return stat.st_mtime, stat.st_size


def _cache_key(source_path: str, embedder_name: str) -> str:
    digest = hashlib.sha256(f"{source_path}::{embedder_name}".encode()).hexdigest()
    return digest[:24]


class EmbeddingCache:
    """Caches per-file chunk embeddings on disk as ``.npy`` + ``.json`` pairs."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _paths_for(self, source_path: str, embedder_name: str) -> tuple[Path, Path]:
        key = _cache_key(source_path, embedder_name)
        return self.cache_dir / f"{key}.npy", self.cache_dir / f"{key}.json"

    def get(
        self,
        source_path: str,
        signature: FileSignature,
        embedder: Embedder,
        chunks: Sequence[Chunk],
    ) -> np.ndarray | None:
        """Return cached embeddings for ``chunks`` if the cache entry is still valid."""
        npy_path, json_path = self._paths_for(source_path, embedder.name)
        if not npy_path.exists() or not json_path.exists():
            return None
        try:
            meta = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        if meta.get("embedder") != embedder.name or meta.get("dim") != embedder.dim:
            return None
        if tuple(meta.get("signature", ())) != tuple(signature):
            return None
        if meta.get("chunk_ids") != [c.id for c in chunks]:
            return None

        try:
            vectors = np.load(npy_path)
        except (OSError, ValueError):
            return None
        if vectors.shape != (len(chunks), embedder.dim):
            return None
        return vectors

    def put(
        self,
        source_path: str,
        signature: FileSignature,
        embedder: Embedder,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """Persist ``vectors`` (aligned with ``chunks``) for ``source_path``."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path, json_path = self._paths_for(source_path, embedder.name)
        np.save(npy_path, vectors)
        meta = {
            "path": source_path,
            "embedder": embedder.name,
            "dim": embedder.dim,
            "signature": list(signature),
            "chunk_ids": [c.id for c in chunks],
        }
        json_path.write_text(json.dumps(meta))

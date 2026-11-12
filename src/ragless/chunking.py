"""Pluggable chunkers that split document text into retrievable :class:`Chunk` objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ragless.types import Chunk


class Chunker(ABC):
    """Base class for splitting a document's text into chunks.

    Implementations must preserve enough information to report the original
    ``start_line``/``end_line`` span for every chunk they produce.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        *,
        path: str | None,
        chunk_id_prefix: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split ``text`` into a list of :class:`Chunk`.

        ``chunk_id_prefix`` should be used to build globally unique chunk ids,
        e.g. ``f"{chunk_id_prefix}:{index}"``.
        """
        raise NotImplementedError


class LineChunker(Chunker):
    """Splits text into fixed-size, overlapping windows of lines.

    This is the default chunker. It is format-agnostic (works for source code,
    markdown, plain text, ...) and always keeps track of the 1-indexed
    ``start_line``/``end_line`` range each chunk came from.
    """

    def __init__(self, lines_per_chunk: int = 40, overlap: int = 5) -> None:
        if lines_per_chunk <= 0:
            raise ValueError("lines_per_chunk must be positive")
        if overlap < 0 or overlap >= lines_per_chunk:
            raise ValueError("overlap must be in [0, lines_per_chunk)")
        self.lines_per_chunk = lines_per_chunk
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        *,
        path: str | None,
        chunk_id_prefix: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        lines = text.splitlines()
        if not lines:
            return []

        meta = dict(metadata or {})
        step = self.lines_per_chunk - self.overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0
        n = len(lines)
        while start < n:
            end = min(start + self.lines_per_chunk, n)
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        id=f"{chunk_id_prefix}:{index}",
                        text=chunk_text,
                        path=path,
                        start_line=start + 1,
                        end_line=end,
                        metadata=meta,
                    )
                )
                index += 1
            if end == n:
                break
            start += step
        return chunks

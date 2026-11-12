"""Core data types shared across ragless."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A unit of retrievable text, tied back to its source location."""

    id: str
    text: str
    path: str | None
    """Source file path, or ``None`` for in-memory text that has no file."""
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A single scored hit returned from :meth:`ragless.Retriever.search`."""

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    """Per-retriever contribution, e.g. ``{"bm25": 0.8, "fuzzy": 0.4}``."""
    graph_reasons: list[str] = field(default_factory=list)
    """Set only for results added by graph expansion (``search(..., graph=True)``):
    a human-readable path of edges back to whichever seed produced this
    result, e.g. ``["auth.md links to retry.md"]``. Empty for ordinary hits."""

    @property
    def path(self) -> str | None:
        return self.chunk.path

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def start_line(self) -> int:
        return self.chunk.start_line

    @property
    def end_line(self) -> int:
        return self.chunk.end_line

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata

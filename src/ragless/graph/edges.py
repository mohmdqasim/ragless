"""The graph's data model: a typed, weighted, explainable edge between two chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """A directed edge between two chunk ids, discovered by one :class:`EdgeBuilder`.

    Direction is the natural reading of ``reason`` (e.g. "a.py imports
    b.py"), but graph traversal treats edges as usable in either direction
    unless told otherwise -- a chunk defining something you're reading about
    is just as relevant as one using it.
    """

    src: str
    dst: str
    type: str
    weight: float
    reason: str

    def reversed(self) -> Edge:
        return Edge(
            src=self.dst, dst=self.src, type=self.type, weight=self.weight, reason=self.reason
        )

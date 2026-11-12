"""Query-time graph expansion: seeds in, scored+explained candidates out.

See DECISIONS.md for the combined-score formula. Short version: an expanded
node's raw score comes from personalized PageRank (restart = normalized seed
scores) or bounded BFS with per-hop decay; then, unless ``keep_seeds=False``,
every non-seed node's raw score is rescaled so the best one never exceeds
the weakest seed's score -- seeds are never outranked by expansion unless
the caller explicitly asks for that by passing ``keep_seeds=False``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ragless.context.passage import Passage
from ragless.context.pipeline import ContextStep, PipelineState
from ragless.graph.edges import Edge
from ragless.graph.pagerank import bounded_bfs, personalized_pagerank

if TYPE_CHECKING:
    from ragless.graph.chunk_graph import ChunkGraph


@dataclass
class ExpandedNode:
    """One node reached by expansion: its score and the path of edges to a seed."""

    chunk_id: str
    score: float
    hops: int
    path: list[Edge] = field(default_factory=list)
    is_seed: bool = False

    @property
    def reason(self) -> str:
        if self.is_seed or not self.path:
            return "direct match"
        return " -> ".join(e.reason for e in self.path)


def expand_from_seeds(
    adjacency: Mapping[str, Sequence[tuple[str, float]]],
    seed_scores: Mapping[str, float],
    *,
    mode: str = "pagerank",
    hops: int = 2,
    max_expanded: int = 20,
    min_score: float = 0.0,
    keep_seeds: bool = True,
    edge_lookup: Mapping[tuple[str, str], Edge] | None = None,
) -> dict[str, ExpandedNode]:
    """Expand ``seed_scores`` across ``adjacency``. Returns ``{chunk_id: ExpandedNode}``.

    Always includes every seed (as an :class:`ExpandedNode` with ``is_seed=True``
    and its original score unchanged), plus up to ``max_expanded`` additional
    nodes above ``min_score``.
    """
    positive_seeds = {k: v for k, v in seed_scores.items() if v > 0}
    if not positive_seeds:
        return {
            k: ExpandedNode(chunk_id=k, score=v, hops=0, is_seed=True)
            for k, v in seed_scores.items()
        }

    paths: dict[str, list[str]] = {}
    if mode == "bfs":
        raw = bounded_bfs(adjacency, positive_seeds, hops=hops, decay=0.6)
        raw_scores = {k: v[0] for k, v in raw.items()}
        paths = {k: v[1] for k, v in raw.items()}
    else:
        raw_scores = personalized_pagerank(adjacency, positive_seeds)
        paths = _pagerank_paths(adjacency, positive_seeds, raw_scores, hops)

    non_seed_scores = {k: v for k, v in raw_scores.items() if k not in positive_seeds and v > 0}
    if keep_seeds and non_seed_scores:
        weakest_seed = min(positive_seeds.values())
        strongest_expanded = max(non_seed_scores.values())
        scale = weakest_seed / strongest_expanded if strongest_expanded > 0 else 1.0
        non_seed_scores = {k: v * scale for k, v in non_seed_scores.items()}

    ranked = sorted(non_seed_scores.items(), key=lambda kv: kv[1], reverse=True)
    nodes: dict[str, ExpandedNode] = {
        chunk_id: ExpandedNode(chunk_id=chunk_id, score=score, hops=0, is_seed=True)
        for chunk_id, score in seed_scores.items()
    }
    kept = 0
    for chunk_id, score in ranked:
        if kept >= max_expanded or score < min_score:
            continue
        node_path = paths.get(chunk_id, [])
        edges = _edges_for_path(node_path, edge_lookup) if edge_lookup else []
        nodes[chunk_id] = ExpandedNode(
            chunk_id=chunk_id, score=score, hops=max(0, len(node_path) - 1), path=edges
        )
        kept += 1
    return nodes


def _pagerank_paths(
    adjacency: Mapping[str, Sequence[tuple[str, float]]],
    seeds: Mapping[str, float],
    scores: Mapping[str, float],
    hops: int,
) -> dict[str, list[str]]:
    """A cheap greedy shortest-ish path for explanation purposes (PageRank itself has no paths).

    Each path includes its originating seed as the first element (so a
    path's length minus one is its hop count), matching :func:`bounded_bfs`'s
    convention.
    """
    from collections import deque

    visited: dict[str, list[str]] = {s: [s] for s in seeds}
    frontier: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    while frontier:
        node, hop = frontier.popleft()
        if hop >= hops:
            continue
        for neighbor, _weight in adjacency.get(node, ()):
            if neighbor in visited:
                continue
            visited[neighbor] = [*visited[node], neighbor]
            frontier.append((neighbor, hop + 1))
    return visited


def _edges_for_path(
    node_path: list[str], edge_lookup: Mapping[tuple[str, str], Edge]
) -> list[Edge]:
    edges: list[Edge] = []
    prev: str | None = None
    for node in node_path:
        if prev is not None:
            edge = edge_lookup.get((prev, node)) or edge_lookup.get((node, prev))
            if edge is not None:
                edges.append(edge)
        prev = node
    return edges


class GraphExpand(ContextStep):
    """Adds graph-expanded passages to the pipeline, ahead of expand/merge/dedup/etc.

    Needs ``state.get_chunk`` to materialize a :class:`Passage` for a chunk
    that was never part of the initial search results; if it's unset (no
    :class:`~ragless.core.Retriever` behind this pipeline), this step is a
    harmless no-op.
    """

    def __init__(
        self,
        chunk_graph: ChunkGraph,
        *,
        mode: str = "pagerank",
        hops: int = 2,
        max_expanded: int = 20,
        min_score: float = 0.0,
        keep_seeds: bool = True,
        edge_weights: dict[str, float] | None = None,
    ) -> None:
        self.chunk_graph = chunk_graph
        self.mode = mode
        self.hops = hops
        self.max_expanded = max_expanded
        self.min_score = min_score
        self.keep_seeds = keep_seeds
        self.edge_weights = edge_weights

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        if state.get_chunk is None or not passages:
            return passages

        by_chunk_id: dict[str, Passage] = {p.chunk_id: p for p in passages if p.chunk_id}
        if not by_chunk_id:
            return passages

        seed_scores = {chunk_id: p.score for chunk_id, p in by_chunk_id.items()}
        nodes = expand_from_seeds(
            self.chunk_graph.adjacency(self.edge_weights),
            seed_scores,
            mode=self.mode,
            hops=self.hops,
            max_expanded=self.max_expanded,
            min_score=self.min_score,
            keep_seeds=self.keep_seeds,
            edge_lookup=self.chunk_graph.edge_lookup(),
        )

        out = list(passages)
        for chunk_id, node in nodes.items():
            if node.is_seed:
                continue
            chunk = state.get_chunk(chunk_id)
            if chunk is None:
                continue
            out.append(
                Passage(
                    text=chunk.text,
                    doc_id=self.chunk_graph.chunk_doc_id.get(chunk_id, chunk_id),
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=node.score,
                    reasons=[f"graph: {node.reason}"],
                    metadata=dict(chunk.metadata),
                    chunk_id=chunk_id,
                )
            )
        return out

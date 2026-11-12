"""Personalized PageRank (NumPy, no SciPy) and a bounded-BFS alternative.

Both operate on a plain adjacency mapping ``{node_id: [(neighbor_id, weight), ...]}``
-- edges are expected to already be symmetrized by the caller (``ChunkGraph``
builds this both ways), so these functions don't care about edge direction.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np

Adjacency = Mapping[str, Sequence[tuple[str, float]]]


def personalized_pagerank(
    adjacency: Adjacency,
    restart: Mapping[str, float],
    *,
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Personalized PageRank with restart, via vectorized NumPy power iteration.

    ``restart`` is normalized to sum to 1 internally. Returns a score per
    node that was reachable from a nonzero-restart node (isolated nodes with
    no restart mass and no incoming edges from the reachable set are simply
    absent, not zero -- callers treat a missing node as score 0).
    """
    nodes = list(adjacency)
    if not nodes:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    restart_total = sum(max(0.0, v) for v in restart.values())
    r = np.zeros(n, dtype=np.float64)
    if restart_total > 0:
        for node, weight in restart.items():
            i = index.get(node)
            if i is not None and weight > 0:
                r[i] = weight / restart_total

    src_idx: list[int] = []
    dst_idx: list[int] = []
    weights: list[float] = []
    out_weight = np.zeros(n, dtype=np.float64)
    for node, neighbors in adjacency.items():
        i = index[node]
        for neighbor, weight in neighbors:
            j = index.get(neighbor)
            if j is None or weight <= 0:
                continue
            src_idx.append(i)
            dst_idx.append(j)
            weights.append(weight)
            out_weight[i] += weight

    src_arr = np.array(src_idx, dtype=np.int64)
    dst_arr = np.array(dst_idx, dtype=np.int64)
    weight_arr = np.array(weights, dtype=np.float64)
    safe_out = np.where(out_weight > 0, out_weight, 1.0)
    norm_weight_arr = weight_arr / safe_out[src_arr] if len(weight_arr) else weight_arr

    p = r.copy()
    dangling = out_weight == 0
    for _ in range(max_iter):
        flow = np.zeros(n, dtype=np.float64)
        if len(src_arr):
            np.add.at(flow, dst_arr, p[src_arr] * norm_weight_arr)
        dangling_mass = p[dangling].sum()
        new_p = alpha * (flow + dangling_mass * r) + (1 - alpha) * r
        if np.abs(new_p - p).sum() < tol:
            p = new_p
            break
        p = new_p

    return {node: float(p[i]) for node, i in index.items()}


def bounded_bfs(
    adjacency: Adjacency,
    seed_scores: Mapping[str, float],
    *,
    hops: int = 2,
    decay: float = 0.6,
) -> dict[str, tuple[float, list[str]]]:
    """Bounded BFS expansion: returns ``{node: (score, path_of_node_ids)}``.

    ``score`` for a reached node is the max, over every discovered path from
    any seed, of ``seed_score * decay**len(path) * product(edge weights)``.
    Seeds themselves are included at hop 0 with their original score, and a
    path containing just themselves (so a path's length minus one is its hop
    count, and the first two elements of any non-seed path give the edge out
    of the seed).
    """
    best: dict[str, tuple[float, list[str]]] = {
        node: (score, [node]) for node, score in seed_scores.items() if score > 0
    }
    frontier: deque[tuple[str, int, float, list[str]]] = deque(
        (node, 0, score, [node]) for node, score in seed_scores.items() if score > 0
    )

    while frontier:
        node, hop, score, path = frontier.popleft()
        if hop >= hops:
            continue
        for neighbor, weight in adjacency.get(node, ()):
            if weight <= 0:
                continue
            candidate_score = score * decay * weight
            new_path = [*path, neighbor]
            current = best.get(neighbor)
            if current is None or candidate_score > current[0]:
                best[neighbor] = (candidate_score, new_path)
                frontier.append((neighbor, hop + 1, candidate_score, new_path))
    return best

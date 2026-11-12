"""ChunkGraph: the built graph over a corpus's chunks, with persistence and incremental update.

See DECISIONS.md ("Edge builders: local vs. global") for exactly what an
incremental :meth:`ChunkGraph.update` does and doesn't re-scan, and why the
result still matches a full rebuild.
"""

from __future__ import annotations

import json
import xml.sax.saxutils as xml_escape
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragless.cache import FileSignature
from ragless.exceptions import MissingDependencyError
from ragless.graph.builders import (
    GLOBAL_BUILDER_NAMES,
    LOCAL_BUILDER_NAMES,
    BuilderResult,
    EdgeBuilder,
    builder_fingerprint,
    make_build_context,
)
from ragless.graph.edges import Edge
from ragless.types import Chunk

GRAPH_CACHE_VERSION = 1


def _could_now_resolve(target: str, candidate_paths: set[str]) -> bool:
    """Conservative (lean-permissive) check: could a newly touched path satisfy
    a previously unresolved reference? False negatives would silently break
    the "incremental == full rebuild" guarantee, so this errs toward
    over-matching (extra rescans cost time, not correctness)."""
    t = target.strip().lstrip("./").rstrip("/").lower()
    if not t:
        return False
    key = t.rsplit("/", 1)[-1]
    key_stem = key.rsplit(".", 1)[0] if "." in key else key
    for p in candidate_paths:
        pl = p.lower()
        if key in pl or (key_stem and key_stem in pl):
            return True
    return False


class ChunkGraph:
    """The built graph: directed, typed, weighted :class:`Edge` objects over chunk ids."""

    def __init__(
        self,
        edges: list[Edge],
        chunk_doc_id: dict[str, str],
        file_signatures: dict[str, FileSignature],
        unresolved: dict[str, dict[str, set[str]]],
        builder_fp: str,
        nodes: list[str] | None = None,
    ) -> None:
        self.edges = edges
        self.chunk_doc_id = chunk_doc_id
        self.file_signatures = file_signatures
        self.unresolved = unresolved
        self.builder_fp = builder_fp
        self._nodes = (
            nodes if nodes is not None else sorted({e.src for e in edges} | {e.dst for e in edges})
        )
        self._adj: dict[str, list[tuple[str, float]]] | None = None
        self._edge_lookup: dict[tuple[str, str], Edge] | None = None

    # -- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        chunk_doc_id: dict[str, str],
        doc_text: dict[str, str],
        builders: Sequence[EdgeBuilder],
        file_signatures: dict[str, FileSignature] | None = None,
    ) -> ChunkGraph:
        ctx = make_build_context(chunks, chunk_doc_id, doc_text)
        edges: list[Edge] = []
        unresolved: dict[str, dict[str, set[str]]] = {}
        for b in builders:
            result: BuilderResult = b.build(ctx)
            edges.extend(result.edges)
            if result.unresolved:
                unresolved[b.name] = result.unresolved
        return cls(
            edges=edges,
            chunk_doc_id=dict(chunk_doc_id),
            file_signatures=dict(file_signatures or {}),
            unresolved=unresolved,
            builder_fp=builder_fingerprint(builders),
            nodes=[c.id for c in chunks],
        )

    def update(
        self,
        chunks: Sequence[Chunk],
        chunk_doc_id: dict[str, str],
        doc_text: dict[str, str],
        builders: Sequence[EdgeBuilder],
        file_signatures: dict[str, FileSignature],
    ) -> ChunkGraph:
        old_paths = set(self.file_signatures)
        new_paths = set(file_signatures)
        added = new_paths - old_paths
        removed = old_paths - new_paths
        changed = {
            p for p in (old_paths & new_paths) if self.file_signatures[p] != file_signatures[p]
        }
        touched = added | removed | changed
        # in-memory text has no (mtime, size) signature to diff, so its content could
        # have silently changed between runs -- always re-treat it as touched rather
        # than risk serving stale edges for it.
        touched |= {d for d in chunk_doc_id.values() if d.startswith("mem:")}
        touched |= {d for d in self.chunk_doc_id.values() if d.startswith("mem:")}

        rescan_extra: set[str] = set()
        for builder_unresolved in self.unresolved.values():
            for src_path, targets in builder_unresolved.items():
                if src_path in touched:
                    continue
                if any(_could_now_resolve(t, added | changed) for t in targets):
                    rescan_extra.add(src_path)

        # A local-builder edge whose far endpoint changed is dropped below (it may be
        # stale), but regenerating it means re-scanning its *source* file again -- the
        # file that changed is not necessarily the one whose content produces the edge.
        rescan_for_dropped = {
            self.chunk_doc_id.get(e.src)
            for e in self.edges
            if e.type in LOCAL_BUILDER_NAMES
            and (self.chunk_doc_id.get(e.src) in touched or self.chunk_doc_id.get(e.dst) in touched)
        }
        rescan_for_dropped.discard(None)

        local_sources = touched | rescan_extra | rescan_for_dropped

        kept_edges = [
            e
            for e in self.edges
            if self.chunk_doc_id.get(e.src) not in touched
            and self.chunk_doc_id.get(e.dst) not in touched
        ]
        kept_edges = [
            e
            for e in kept_edges
            if not (
                e.type in LOCAL_BUILDER_NAMES and (self.chunk_doc_id.get(e.src) in local_sources)
            )
        ]

        new_edges: list[Edge] = list(kept_edges)
        new_unresolved: dict[str, dict[str, set[str]]] = {
            name: {p: t for p, t in d.items() if p not in local_sources}
            for name, d in self.unresolved.items()
        }

        ctx_local = make_build_context(
            chunks, chunk_doc_id, doc_text, only_source_paths=local_sources
        )
        ctx_global = make_build_context(chunks, chunk_doc_id, doc_text)

        for b in builders:
            if b.name in GLOBAL_BUILDER_NAMES:
                result = b.build(ctx_global)
                new_edges = [e for e in new_edges if e.type != b.name] + result.edges
                new_unresolved[b.name] = result.unresolved
            elif local_sources:
                result = b.build(ctx_local)
                new_edges.extend(result.edges)
                new_unresolved.setdefault(b.name, {}).update(result.unresolved)

        return ChunkGraph(
            edges=new_edges,
            chunk_doc_id=dict(chunk_doc_id),
            file_signatures=dict(file_signatures),
            unresolved=new_unresolved,
            builder_fp=builder_fingerprint(builders),
            nodes=[c.id for c in chunks],
        )

    # -- traversal helpers used by graph expansion -------------------------

    def adjacency(
        self, edge_weights: dict[str, float] | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """Symmetrized adjacency. ``edge_weights`` (edge type -> multiplier) reweights
        at query time without touching the stored, builder-assigned weights."""
        if edge_weights is None:
            if self._adj is None:
                self._adj = self._build_adjacency(None)
            return self._adj
        return self._build_adjacency(edge_weights)

    def _build_adjacency(
        self, edge_weights: dict[str, float] | None
    ) -> dict[str, list[tuple[str, float]]]:
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for e in self.edges:
            w = e.weight * (edge_weights.get(e.type, 1.0) if edge_weights else 1.0)
            if w <= 0:
                continue
            adj[e.src].append((e.dst, w))
            adj[e.dst].append((e.src, w))
        for node in self._nodes:
            adj.setdefault(node, [])
        return dict(adj)

    def edge_lookup(self) -> dict[tuple[str, str], Edge]:
        if self._edge_lookup is None:
            lookup: dict[tuple[str, str], Edge] = {}
            for e in self.edges:
                lookup.setdefault((e.src, e.dst), e)
            self._edge_lookup = lookup
        return self._edge_lookup

    # -- public query API ---------------------------------------------------

    def neighbors(
        self, chunk_id: str, *, edge_types: Sequence[str] | None = None, direction: str = "both"
    ) -> list[tuple[str, Edge]]:
        out: list[tuple[str, Edge]] = []
        for e in self.edges:
            if edge_types and e.type not in edge_types:
                continue
            if direction in ("both", "out") and e.src == chunk_id:
                out.append((e.dst, e))
            if direction in ("both", "in") and e.dst == chunk_id:
                out.append((e.src, e))
        return out

    def explain(self, chunk_id: str) -> list[Edge]:
        """Every edge touching ``chunk_id``, in either direction."""
        return [e for e in self.edges if e.src == chunk_id or e.dst == chunk_id]

    def related(self, path: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-``k`` other file paths related to ``path``, by aggregated edge weight."""
        scores: dict[str, float] = defaultdict(float)
        in_file = {cid for cid, doc in self.chunk_doc_id.items() if doc == path}
        for e in self.edges:
            if e.src in in_file and self.chunk_doc_id.get(e.dst) != path:
                scores[self.chunk_doc_id[e.dst]] += e.weight
            elif e.dst in in_file and self.chunk_doc_id.get(e.src) != path:
                scores[self.chunk_doc_id[e.src]] += e.weight
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def stats(self) -> dict[str, Any]:
        by_type = Counter(e.type for e in self.edges)
        return {"nodes": len(self._nodes), "edges": len(self.edges), "edges_by_type": dict(by_type)}

    # -- export ---------------------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "nodes": self._nodes,
            "edges": [
                {"src": e.src, "dst": e.dst, "type": e.type, "weight": e.weight, "reason": e.reason}
                for e in self.edges
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def to_graphml(self) -> str:
        """Hand-rolled GraphML (plain XML) -- no extra dependency needed to produce it."""

        # quoteattr() returns the *whole* delimited fragment (e.g. '"x"' or "'x'",
        # picking whichever quote char avoids escaping) -- never strip and rewrap it
        # in a hardcoded quote, or a value containing the other quote char breaks.
        esc_attr = xml_escape.quoteattr
        esc_text = xml_escape.escape
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '<key id="etype" for="edge" attr.name="type" attr.type="string"/>',
            '<key id="weight" for="edge" attr.name="weight" attr.type="double"/>',
            '<key id="reason" for="edge" attr.name="reason" attr.type="string"/>',
            '<graph edgedefault="directed">',
        ]
        for n in self._nodes:
            lines.append(f"<node id={esc_attr(n)}/>")
        for i, e in enumerate(self.edges):
            lines.append(
                f"<edge id={esc_attr(f'e{i}')} source={esc_attr(e.src)} target={esc_attr(e.dst)}>"
                f'<data key="etype">{esc_text(e.type)}</data>'
                f'<data key="weight">{e.weight}</data>'
                f'<data key="reason">{esc_text(e.reason)}</data>'
                "</edge>"
            )
        lines.append("</graph></graphml>")
        return "\n".join(lines)

    def to_networkx(self) -> Any:
        """Requires ``pip install ragless[graph]`` (networkx); an optional convenience."""
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - exercised without extra installed
            raise MissingDependencyError("networkx export", "graph") from exc
        g = nx.DiGraph()
        g.add_nodes_from(self._nodes)
        for e in self.edges:
            g.add_edge(e.src, e.dst, type=e.type, weight=e.weight, reason=e.reason)
        return g

    # -- persistence ----------------------------------------------------------

    def save(self, cache_dir: str | Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GRAPH_CACHE_VERSION,
            "builder_fp": self.builder_fp,
            "nodes": self._nodes,
            "edges": [[e.src, e.dst, e.type, e.weight, e.reason] for e in self.edges],
            "chunk_doc_id": self.chunk_doc_id,
            "file_signatures": {p: list(sig) for p, sig in self.file_signatures.items()},
            "unresolved": {
                builder: {p: sorted(t) for p, t in d.items()}
                for builder, d in self.unresolved.items()
            },
        }
        (cache_dir / "graph.json").write_text(json.dumps(payload))

    @classmethod
    def load(cls, cache_dir: str | Path) -> ChunkGraph | None:
        path = Path(cache_dir) / "graph.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != GRAPH_CACHE_VERSION:
            return None
        try:
            edges = [
                Edge(src=s, dst=d, type=t, weight=w, reason=r) for s, d, t, w, r in payload["edges"]
            ]
            file_signatures = {p: tuple(sig) for p, sig in payload["file_signatures"].items()}
            unresolved = {
                builder: {p: set(t) for p, t in d.items()}
                for builder, d in payload.get("unresolved", {}).items()
            }
            return cls(
                edges=edges,
                chunk_doc_id=payload["chunk_doc_id"],
                file_signatures=file_signatures,
                unresolved=unresolved,
                builder_fp=payload["builder_fp"],
                nodes=payload["nodes"],
            )
        except (KeyError, ValueError, TypeError):
            return None


def build_or_update_graph(
    cache_dir: str | Path | None,
    chunks: Sequence[Chunk],
    chunk_doc_id: dict[str, str],
    doc_text: dict[str, str],
    builders: Sequence[EdgeBuilder],
    file_signatures: dict[str, FileSignature],
) -> ChunkGraph:
    """Build fresh, or load+incrementally update the cached graph in ``cache_dir``."""
    fp = builder_fingerprint(builders)
    cached = ChunkGraph.load(cache_dir) if cache_dir else None
    if cached is not None and cached.builder_fp == fp:
        graph = cached.update(chunks, chunk_doc_id, doc_text, builders, file_signatures)
    else:
        graph = ChunkGraph.build(chunks, chunk_doc_id, doc_text, builders, file_signatures)
    if cache_dir:
        graph.save(cache_dir)
    return graph

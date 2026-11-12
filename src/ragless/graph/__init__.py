"""ragless.graph: expand retrieval using structure already in the files.

No LLM is involved anywhere here -- edges come from links, imports, symbol
definitions/usages, heading structure, shared rare terms, and (only with the
``semantic`` extra) embedding similarity. Ships in the base package with no
new required dependencies; ``pip install ragless[graph]`` only adds
``networkx`` for :meth:`ChunkGraph.to_networkx`.
"""

from __future__ import annotations

from ragless.graph.builders import (
    EdgeBuilder,
    ImportEdges,
    LinkEdges,
    SemanticKNNEdges,
    SharedTermEdges,
    StructureEdges,
    SymbolEdges,
    build_default_edge_builders,
)
from ragless.graph.chunk_graph import ChunkGraph, build_or_update_graph
from ragless.graph.edges import Edge
from ragless.graph.expand import ExpandedNode, GraphExpand, expand_from_seeds
from ragless.graph.pagerank import bounded_bfs, personalized_pagerank

__all__ = [
    "Edge",
    "EdgeBuilder",
    "LinkEdges",
    "ImportEdges",
    "SymbolEdges",
    "StructureEdges",
    "SharedTermEdges",
    "SemanticKNNEdges",
    "build_default_edge_builders",
    "ChunkGraph",
    "build_or_update_graph",
    "ExpandedNode",
    "GraphExpand",
    "expand_from_seeds",
    "personalized_pagerank",
    "bounded_bfs",
]

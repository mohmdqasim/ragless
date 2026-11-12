"""ragless: file-based retrieval for RAG, no vector database required."""

from ragless.chunking import Chunker, LineChunker
from ragless.context import (
    Citation,
    Compress,
    Context,
    ContextStep,
    Dedup,
    Diversify,
    MergeAdjacent,
    Pack,
    Passage,
    PipelineState,
    TiktokenCounter,
    WindowExpand,
    build_default_pipeline,
    estimate_tokens,
)
from ragless.core import Retriever
from ragless.embedders.base import Embedder
from ragless.exceptions import MissingDependencyError, RaglessError, UnknownRetrieverError
from ragless.fusion import reciprocal_rank_fusion
from ragless.graph import (
    ChunkGraph,
    Edge,
    EdgeBuilder,
    ExpandedNode,
    GraphExpand,
    ImportEdges,
    LinkEdges,
    SemanticKNNEdges,
    SharedTermEdges,
    StructureEdges,
    SymbolEdges,
    build_default_edge_builders,
)
from ragless.rerank import CrossEncoderReranker, Reranker
from ragless.retrievers import BaseRetriever, BM25Retriever, FuzzyRetriever, SemanticRetriever
from ragless.types import Chunk, SearchResult

__version__ = "0.3.0"

__all__ = [
    "Retriever",
    "Chunk",
    "SearchResult",
    "Chunker",
    "LineChunker",
    "BaseRetriever",
    "BM25Retriever",
    "FuzzyRetriever",
    "SemanticRetriever",
    "Embedder",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "Reranker",
    "RaglessError",
    "MissingDependencyError",
    "UnknownRetrieverError",
    "Context",
    "Citation",
    "Passage",
    "ContextStep",
    "WindowExpand",
    "MergeAdjacent",
    "Dedup",
    "Diversify",
    "Compress",
    "Pack",
    "PipelineState",
    "build_default_pipeline",
    "estimate_tokens",
    "TiktokenCounter",
    "ChunkGraph",
    "Edge",
    "EdgeBuilder",
    "LinkEdges",
    "ImportEdges",
    "SymbolEdges",
    "StructureEdges",
    "SharedTermEdges",
    "SemanticKNNEdges",
    "build_default_edge_builders",
    "GraphExpand",
    "ExpandedNode",
    "__version__",
]

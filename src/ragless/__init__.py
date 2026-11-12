"""ragless: file-based retrieval for RAG, no vector database required."""

from ragless.chunking import Chunker, LineChunker
from ragless.core import Retriever
from ragless.embedders.base import Embedder
from ragless.exceptions import MissingDependencyError, RaglessError, UnknownRetrieverError
from ragless.fusion import reciprocal_rank_fusion
from ragless.rerank import CrossEncoderReranker, Reranker
from ragless.retrievers import BaseRetriever, BM25Retriever, FuzzyRetriever, SemanticRetriever
from ragless.types import Chunk, SearchResult

__version__ = "0.1.0"

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
    "__version__",
]

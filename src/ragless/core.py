"""The library's main entry point: :class:`Retriever`."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ragless.cache import EmbeddingCache, FileSignature, file_signature
from ragless.chunking import Chunker, LineChunker
from ragless.embedders.base import Embedder
from ragless.exceptions import UnknownRetrieverError
from ragless.fusion import reciprocal_rank_fusion
from ragless.rerank import Reranker
from ragless.retrievers.base import BaseRetriever
from ragless.retrievers.bm25 import BM25Retriever
from ragless.retrievers.fuzzy import FuzzyRetriever
from ragless.retrievers.semantic import SemanticRetriever
from ragless.types import Chunk, SearchResult

DEFAULT_ENGINES: tuple[str, ...] = ("bm25", "fuzzy")

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".ragless_cache",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    "dist",
    "build",
}

MetadataSource = Mapping[str, Any] | Callable[[str | None], Mapping[str, Any]] | None
EngineSpec = Sequence[str] | Mapping[str, BaseRetriever] | None
Filters = Mapping[str, Any | Callable[[Any], bool]] | None
PathFilter = str | Sequence[str] | None


class Retriever:
    """Index a directory (or in-memory text) and search it with no vector DB.

    Three retrieval strategies -- BM25 keyword search, fuzzy string matching,
    and (optionally) semantic embedding search -- are combined with
    Reciprocal Rank Fusion. Everything is pluggable: chunker, retrievers,
    and embedder can all be swapped out.

    Examples:
        >>> r = Retriever("./docs")
        >>> results = r.search("installation steps", mode="hybrid")

        >>> r = Retriever(["first note", "second note about pandas"])
        >>> results = r.search("pandas")
    """

    def __init__(
        self,
        source: str | Path | Sequence[str] | None = None,
        *,
        texts: Sequence[str] | None = None,
        chunker: Chunker | None = None,
        engines: EngineSpec = None,
        embedder: Embedder | None = None,
        cache_dir: str | Path | None = None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        metadata: MetadataSource = None,
        encoding: str = "utf-8",
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """
        Args:
            source: A directory path, a single file path, or a sequence of
                raw in-memory strings (each treated as one document).
            texts: Additional in-memory strings to index alongside ``source``.
            chunker: Splits document text into :class:`~ragless.types.Chunk`.
                Defaults to :class:`~ragless.chunking.LineChunker`.
            engines: Which retrievers to build and query. Either a sequence
                of names from ``{"bm25", "fuzzy", "semantic"}`` (default
                ``("bm25", "fuzzy")``), or a mapping of name to a pre-built
                :class:`~ragless.retrievers.base.BaseRetriever` for full
                control over pluggable retrievers.
            embedder: Embedding backend used by the ``"semantic"`` engine.
                Defaults to :class:`~ragless.embedders.fastembed_embedder.FastEmbedEmbedder`,
                which requires ``pip install ragless[semantic]``.
            cache_dir: Directory to store the semantic embedding cache
                (sidecar ``.npy``/``.json`` files). Only files on disk are
                cached; in-memory text is always re-embedded.
            include: Glob patterns (matched against each file's relative
                POSIX path); only matching files are indexed.
            exclude: Glob patterns for files to skip.
            metadata: Either a constant mapping applied to every chunk, or a
                callable ``path -> mapping`` for per-document metadata (used
                by ``filters=`` in :meth:`search`). ``path`` is ``None`` for
                in-memory text.
            encoding: Text encoding used to read files. Files that fail to
                decode, or contain null bytes, are treated as binary and
                skipped.
            rrf_k: The ``k`` constant in Reciprocal Rank Fusion.
            weights: Optional per-engine weights used during fusion, e.g.
                ``{"bm25": 1.0, "semantic": 2.0}``.
        """
        self.chunker = chunker or LineChunker()
        self.rrf_k = rrf_k
        self.weights = weights
        self._cache_dir = cache_dir
        self._reranker: Reranker | None = None

        documents = _load_documents(
            source, texts, include=include, exclude=exclude, encoding=encoding
        )

        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[str, Chunk] = {}
        self._file_signatures: dict[str, FileSignature] = {}

        mem_index = 0
        for path, text in documents:
            if path is None:
                prefix = f"mem:{mem_index}"
                mem_index += 1
            else:
                prefix = path
                self._file_signatures[path] = file_signature(Path(path))

            chunk_metadata = _resolve_metadata(metadata, path)
            for chunk in self.chunker.chunk(
                text, path=path, chunk_id_prefix=prefix, metadata=chunk_metadata
            ):
                self._chunks.append(chunk)
                self._chunk_by_id[chunk.id] = chunk

        self._retrievers: dict[str, BaseRetriever] = {}
        if isinstance(engines, Mapping):
            self._retrievers.update(engines)
        else:
            for name in engines if engines is not None else DEFAULT_ENGINES:
                self._retrievers[name] = self._build_engine(name, embedder, cache_dir)

        for _name, retriever in self._retrievers.items():
            if isinstance(retriever, SemanticRetriever):
                retriever.index_with_signatures(self._chunks, self._file_signatures)
            else:
                retriever.index(self._chunks)

    def _build_engine(
        self, name: str, embedder: Embedder | None, cache_dir: str | Path | None
    ) -> BaseRetriever:
        if name == "bm25":
            return BM25Retriever()
        if name == "fuzzy":
            return FuzzyRetriever()
        if name == "semantic":
            resolved_embedder = embedder or _default_embedder()
            cache = EmbeddingCache(cache_dir) if cache_dir is not None else None
            return SemanticRetriever(resolved_embedder, cache=cache)
        raise UnknownRetrieverError(
            f"Unknown retriever engine {name!r}; expected one of "
            f"'bm25', 'fuzzy', 'semantic', or an engines= mapping with a custom retriever."
        )

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 10,
        filters: Filters = None,
        path_filter: PathFilter = None,
        rerank: bool | Reranker = False,
        fetch_k: int | None = None,
    ) -> list[SearchResult]:
        """Search the indexed corpus.

        Args:
            query: The search query.
            mode: ``"hybrid"`` fuses every configured engine; or the name of
                a single configured engine (e.g. ``"bm25"``).
            top_k: Maximum number of results to return.
            filters: Mapping of metadata key to either an exact value or a
                predicate ``value -> bool``; a chunk must satisfy all of them.
            path_filter: One or more glob patterns; a chunk's path must match
                at least one. In-memory chunks (``path is None``) never match.
            rerank: ``True`` to rerank fused candidates with the default
                :class:`~ragless.rerank.CrossEncoderReranker` (requires
                ``pip install ragless[semantic]``), or a custom object
                implementing the :class:`~ragless.rerank.Reranker` protocol.
            fetch_k: How many results to fetch from each engine before fusion
                and filtering. Defaults to ``max(top_k * 5, 25)``.

        Returns:
            Up to ``top_k`` :class:`~ragless.types.SearchResult`, best first.
        """
        if not self._chunks or not query or not query.strip():
            return []

        engine_names = self._resolve_mode(mode)
        effective_fetch_k = fetch_k if fetch_k is not None else max(top_k * 5, 25)

        rankings = {}
        for name in engine_names:
            rankings[name] = self._retrievers[name].query(query, effective_fetch_k)

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights)
        candidates = [
            (chunk_id, score, raw)
            for chunk_id, score, raw in fused
            if self._passes_filters(self._chunk_by_id[chunk_id], filters, path_filter)
        ]

        if rerank:
            return self._rerank(query, candidates, rerank, top_k)

        return [
            SearchResult(chunk=self._chunk_by_id[chunk_id], score=score, scores=raw)
            for chunk_id, score, raw in candidates[:top_k]
        ]

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, dict[str, float]]],
        rerank: bool | Reranker,
        top_k: int,
    ) -> list[SearchResult]:
        reranker = rerank if hasattr(rerank, "rerank") else self._get_default_reranker()
        pool = candidates[: max(top_k * 3, 20)]
        pool_chunks = [self._chunk_by_id[chunk_id] for chunk_id, _, _ in pool]
        raw_by_id = {chunk_id: raw for chunk_id, _, raw in pool}

        reranked = reranker.rerank(query, pool_chunks, top_k=top_k)
        return [
            SearchResult(
                chunk=self._chunk_by_id[chunk_id],
                score=score,
                scores={**raw_by_id.get(chunk_id, {}), "rerank": score},
            )
            for chunk_id, score in reranked
        ]

    def _get_default_reranker(self) -> Reranker:
        if self._reranker is None:
            from ragless.rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _resolve_mode(self, mode: str) -> list[str]:
        if mode == "hybrid":
            return list(self._retrievers.keys())
        if mode in self._retrievers:
            return [mode]
        raise UnknownRetrieverError(
            f"Unknown search mode {mode!r}; configured engines are "
            f"{list(self._retrievers.keys())} (plus 'hybrid')."
        )

    def _passes_filters(self, chunk: Chunk, filters: Filters, path_filter: PathFilter) -> bool:
        if filters:
            for key, expected in filters.items():
                actual = chunk.metadata.get(key)
                if callable(expected):
                    if not expected(actual):
                        return False
                elif actual != expected:
                    return False
        if path_filter:
            if chunk.path is None:
                return False
            patterns = [path_filter] if isinstance(path_filter, str) else list(path_filter)
            if not any(fnmatch.fnmatch(chunk.path, pattern) for pattern in patterns):
                return False
        return True

    def __len__(self) -> int:
        return len(self._chunks)


def _resolve_metadata(metadata: MetadataSource, path: str | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if callable(metadata):
        return dict(metadata(path) or {})
    return dict(metadata)


def _default_embedder() -> Embedder:
    from ragless.embedders.fastembed_embedder import FastEmbedEmbedder

    return FastEmbedEmbedder()


def _is_text_sequence(source: Any) -> bool:
    return isinstance(source, (list, tuple)) and not isinstance(source, (str, bytes))


def _read_text_file(path: Path, encoding: str) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        return None


def _iter_files(
    root: Path, include: Sequence[str] | None, exclude: Sequence[str] | None
) -> list[Path]:
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            rel = file_path.relative_to(root).as_posix()
            if include and not any(fnmatch.fnmatch(rel, pattern) for pattern in include):
                continue
            if exclude and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
                continue
            matches.append(file_path)
    return matches


def _load_documents(
    source: str | Path | Sequence[str] | None,
    texts: Sequence[str] | None,
    *,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    encoding: str,
) -> list[tuple[str | None, str]]:
    documents: list[tuple[str | None, str]] = []

    if source is not None:
        if _is_text_sequence(source):
            documents.extend((None, t) for t in source)
        else:
            path = Path(source)
            if path.is_file():
                content = _read_text_file(path, encoding)
                if content is not None:
                    documents.append((str(path), content))
            elif path.is_dir():
                for file_path in _iter_files(path, include, exclude):
                    content = _read_text_file(file_path, encoding)
                    if content is not None:
                        documents.append((str(file_path), content))
            else:
                raise FileNotFoundError(f"Source path does not exist: {path}")

    if texts:
        documents.extend((None, t) for t in texts)

    return documents

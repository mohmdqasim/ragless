"""The library's main entry point: :class:`Retriever`."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ragless.cache import EmbeddingCache, FileSignature, file_signature
from ragless.chunking import Chunker, LineChunker
from ragless.context import (
    Context,
    ContextStep,
    build_context,
    build_default_pipeline,
    render_prompt,
)
from ragless.context.query_expansion import expand_query_terms, merge_search_results
from ragless.embedders.base import Embedder
from ragless.exceptions import UnknownRetrieverError
from ragless.fusion import reciprocal_rank_fusion
from ragless.graph import (
    ChunkGraph,
    EdgeBuilder,
    ExpandedNode,
    GraphExpand,
    build_default_edge_builders,
    build_or_update_graph,
    expand_from_seeds,
)
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
        edge_builders: Sequence[EdgeBuilder] | None = None,
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
            edge_builders: Custom :class:`~ragless.graph.EdgeBuilder` list for
                the :attr:`graph` property, replacing the default set
                (links, imports, symbols, structure, shared terms, and --
                when a semantic embedder is configured -- semantic kNN).
        """
        self.chunker = chunker or LineChunker()
        self.rrf_k = rrf_k
        self.weights = weights
        self._cache_dir = cache_dir
        self._reranker: Reranker | None = None
        self._edge_builders_override = edge_builders
        self._chunk_graph: ChunkGraph | None = None

        documents = _load_documents(
            source, texts, include=include, exclude=exclude, encoding=encoding
        )

        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[str, Chunk] = {}
        self._file_signatures: dict[str, FileSignature] = {}
        self._doc_text: dict[str, str] = {}
        self._chunk_doc_id: dict[str, str] = {}

        mem_index = 0
        for path, text in documents:
            if path is None:
                prefix = f"mem:{mem_index}"
                mem_index += 1
            else:
                prefix = path
                self._file_signatures[path] = file_signature(Path(path))

            self._doc_text[prefix] = text
            chunk_metadata = _resolve_metadata(metadata, path)
            for chunk in self.chunker.chunk(
                text, path=path, chunk_id_prefix=prefix, metadata=chunk_metadata
            ):
                self._chunks.append(chunk)
                self._chunk_by_id[chunk.id] = chunk
                self._chunk_doc_id[chunk.id] = prefix

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
        graph: bool = False,
        hops: int = 2,
        max_expanded: int = 20,
        graph_mode: str = "pagerank",
        edge_weights: dict[str, float] | None = None,
        min_score: float = 0.0,
        keep_seeds: bool = True,
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
            graph: Expand candidates using :attr:`graph` -- following links,
                imports, symbol references, shared terms, etc. to pull in
                chunks the query never mentioned. Off by default.
            hops: Maximum traversal distance from a seed.
            max_expanded: Maximum number of additional (non-seed) results
                graph expansion can add.
            graph_mode: ``"pagerank"`` (personalized PageRank with restart)
                or ``"bfs"`` (bounded breadth-first search with per-hop decay).
            edge_weights: Per-edge-type multiplier, e.g. ``{"link": 2.0}``,
                applied at query time without rebuilding the graph.
            min_score: Minimum expansion score for a node to be included.
            keep_seeds: If ``True`` (default), expansion can never outrank a
                seed -- the strongest non-seed result is capped at the
                weakest seed's score. ``False`` lets expanded results compete
                purely on score, so a seed could end up ranked lower or cut.

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

        graph_reasons: dict[str, list[str]] = {}
        if graph and candidates:
            candidates, graph_reasons = self._expand_with_graph(
                candidates,
                hops=hops,
                max_expanded=max_expanded,
                mode=graph_mode,
                edge_weights=edge_weights,
                min_score=min_score,
                keep_seeds=keep_seeds,
            )

        if rerank:
            return self._rerank(query, candidates, rerank, top_k, graph_reasons=graph_reasons)

        return [
            SearchResult(
                chunk=self._chunk_by_id[chunk_id],
                score=score,
                scores=raw,
                graph_reasons=graph_reasons.get(chunk_id, []),
            )
            for chunk_id, score, raw in candidates[:top_k]
        ]

    def _expand_with_graph(
        self,
        candidates: list[tuple[str, float, dict[str, float]]],
        *,
        hops: int,
        max_expanded: int,
        mode: str,
        edge_weights: dict[str, float] | None,
        min_score: float,
        keep_seeds: bool,
    ) -> tuple[list[tuple[str, float, dict[str, float]]], dict[str, list[str]]]:
        seed_scores = {chunk_id: score for chunk_id, score, _raw in candidates}
        nodes: dict[str, ExpandedNode] = expand_from_seeds(
            self.graph.adjacency(edge_weights),
            seed_scores,
            mode=mode,
            hops=hops,
            max_expanded=max_expanded,
            min_score=min_score,
            keep_seeds=keep_seeds,
            edge_lookup=self.graph.edge_lookup(),
        )
        existing_ids = {chunk_id for chunk_id, _, _ in candidates}
        merged = list(candidates)
        reasons: dict[str, list[str]] = {}
        for chunk_id, node in nodes.items():
            if node.is_seed or chunk_id not in self._chunk_by_id:
                continue
            reasons[chunk_id] = [node.reason]
            if chunk_id not in existing_ids:
                merged.append((chunk_id, node.score, {"graph": node.score}))
        merged.sort(key=lambda c: c[1], reverse=True)
        return merged, reasons

    def context(
        self,
        query: str,
        *,
        max_tokens: int = 2000,
        mode: str = "hybrid",
        top_k: int | None = None,
        filters: Filters = None,
        path_filter: PathFilter = None,
        rerank: bool | Reranker = False,
        fetch_k: int | None = None,
        window: int = 2,
        merge_gap: int = 0,
        dedup_threshold: float | None = 92.0,
        diversify: bool = True,
        mmr_lambda: float = 0.5,
        compress: bool = False,
        compress_ratio: float = 0.5,
        highlight: bool = True,
        token_counter: Callable[[str], int] | None = None,
        pipeline: Sequence[ContextStep] | None = None,
        query_expansion: bool = False,
        graph: bool = False,
        hops: int = 2,
        max_expanded: int = 20,
        graph_mode: str = "pagerank",
        edge_weights: dict[str, float] | None = None,
        min_score: float = 0.0,
        keep_seeds: bool = True,
    ) -> Context:
        """Search, then assemble ready-to-use context fitted to a token budget.

        Runs :meth:`search` (optionally with RM3-style query expansion), then
        a pipeline of expand -> merge -> dedup -> diversify -> compress ->
        pack steps, and returns a :class:`~ragless.context.Context` with
        ``.passages``, ``.citations``, ``.token_count``, and
        ``.to_text()``/``.to_markdown()``/``.to_xml()``/``.to_json()``.

        Args:
            max_tokens: Token budget for the packed context.
            top_k: Candidate pool size before pipeline steps run. Defaults to
                a generous pool sized off ``max_tokens``.
            window: Neighboring lines to include around each hit.
            merge_gap: Extra line gap still considered "adjacent" when
                merging same-file hits.
            dedup_threshold: Fuzzy-ratio threshold (0-100) above which two
                passages are considered duplicates; ``None``/``0`` disables.
            diversify: Reorder candidates by Maximal Marginal Relevance.
            mmr_lambda: Relevance-vs-diversity trade-off for ``diversify``.
            compress: Extractively compress each passage to its most
                query-relevant sentences/lines.
            compress_ratio: Fraction of sentences/lines kept when
                ``compress=True``.
            highlight: Record character offsets of matched query terms on
                each passage.
            token_counter: Custom token counter, e.g.
                :class:`ragless.context.TiktokenCounter`. Defaults to a
                dependency-free character-based estimate.
            pipeline: A full custom list of
                :class:`~ragless.context.ContextStep`, replacing the default
                pipeline entirely (``window``/``merge_gap``/etc. are then
                ignored).
            query_expansion: Pseudo-relevance feedback: re-search with terms
                pulled from the first pass's top results, off by default.
            graph: Prepend a :class:`~ragless.graph.GraphExpand` step (see
                :attr:`graph`) ahead of the default pipeline, so expanded
                passages also get windowed/merged/deduped/diversified/
                compressed/packed just like any other passage. Has no effect
                if ``pipeline=`` is given explicitly -- add
                ``GraphExpand(r.graph, ...)`` to your own list instead.
            hops, max_expanded, graph_mode, edge_weights, min_score,
                keep_seeds: Forwarded to :class:`~ragless.graph.GraphExpand`
                when ``graph=True``; see :meth:`search` for what each means.

        See :meth:`search` for ``mode``, ``filters``, ``path_filter``,
        ``rerank``, and ``fetch_k``.
        """
        pool_k = top_k if top_k is not None else max(50, max_tokens // 20)
        results = self.search(
            query,
            mode=mode,
            top_k=pool_k,
            filters=filters,
            path_filter=path_filter,
            rerank=False if query_expansion else rerank,
            fetch_k=fetch_k,
        )

        if query_expansion:
            terms = expand_query_terms([r.text for r in results], query)
            if terms:
                expanded = self.search(
                    f"{query} {' '.join(terms)}",
                    mode=mode,
                    top_k=pool_k,
                    filters=filters,
                    path_filter=path_filter,
                    fetch_k=fetch_k,
                )
                results = merge_search_results(results, expanded)
            if rerank:
                results = self._rerank_results(query, results, rerank, pool_k)

        effective_pipeline = list(pipeline) if pipeline is not None else None
        if graph and effective_pipeline is None:
            effective_pipeline = [
                GraphExpand(
                    self.graph,
                    mode=graph_mode,
                    hops=hops,
                    max_expanded=max_expanded,
                    min_score=min_score,
                    keep_seeds=keep_seeds,
                    edge_weights=edge_weights,
                ),
                *build_default_pipeline(
                    window=window,
                    merge_gap=merge_gap,
                    dedup_threshold=dedup_threshold,
                    diversify=diversify,
                    mmr_lambda=mmr_lambda,
                    compress=compress,
                    compress_ratio=compress_ratio,
                ),
            ]

        return build_context(
            query=query,
            results=results,
            doc_text=lambda doc_id: self._doc_text.get(doc_id, ""),
            chunk_doc_id=lambda chunk_id: self._chunk_doc_id.get(chunk_id, chunk_id),
            max_tokens=max_tokens,
            token_counter=token_counter,
            pipeline=effective_pipeline,
            embed_fn=self._embed_fn_for_diversify(),
            get_chunk=lambda chunk_id: self._chunk_by_id.get(chunk_id),
            highlight=highlight,
            window=window,
            merge_gap=merge_gap,
            dedup_threshold=dedup_threshold,
            diversify=diversify,
            mmr_lambda=mmr_lambda,
            compress=compress,
            compress_ratio=compress_ratio,
        )

    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        rerank: bool | Reranker,
        top_k: int,
    ) -> list[SearchResult]:
        reranker = rerank if hasattr(rerank, "rerank") else self._get_default_reranker()
        chunks = [r.chunk for r in results]
        raw_by_id = {r.chunk.id: r.scores for r in results}
        reranked = reranker.rerank(query, chunks, top_k=top_k)
        return [
            SearchResult(
                chunk=self._chunk_by_id[chunk_id],
                score=score,
                scores={**raw_by_id.get(chunk_id, {}), "rerank": score},
            )
            for chunk_id, score in reranked
        ]

    def _embed_fn_for_diversify(self) -> Callable[[Sequence[str]], Any] | None:
        embedder = self._semantic_embedder()
        return embedder.embed if embedder is not None else None

    def _semantic_embedder(self) -> Embedder | None:
        semantic = self._retrievers.get("semantic")
        if semantic is None or not hasattr(semantic, "embedder"):
            return None
        return semantic.embedder

    @property
    def graph(self) -> ChunkGraph:
        """The :class:`~ragless.graph.ChunkGraph` over this corpus.

        Built (or loaded + incrementally updated from ``cache_dir``, if one
        was given) on first access and cached on the instance. Edge builders
        default to :func:`~ragless.graph.build_default_edge_builders`
        (overridable via the ``edge_builders=`` constructor argument).
        """
        if self._chunk_graph is None:
            builders = (
                self._edge_builders_override
                if self._edge_builders_override is not None
                else build_default_edge_builders(self._semantic_embedder())
            )
            self._chunk_graph = build_or_update_graph(
                self._cache_dir,
                self._chunks,
                self._chunk_doc_id,
                self._doc_text,
                builders,
                self._file_signatures,
            )
        return self._chunk_graph

    def prompt(
        self,
        query: str,
        *,
        template: str | Callable[[str, Context], str] | None = None,
        context_format: str = "markdown",
        **context_kwargs: Any,
    ) -> str:
        """Render a ready-to-send prompt string: :meth:`context` plus a template.

        ``ragless`` never calls an LLM itself; this only builds the prompt
        string for whichever client you call yourself.
        """
        ctx = self.context(query, **context_kwargs)
        return render_prompt(query, ctx, template=template, context_format=context_format)

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, dict[str, float]]],
        rerank: bool | Reranker,
        top_k: int,
        graph_reasons: dict[str, list[str]] | None = None,
    ) -> list[SearchResult]:
        graph_reasons = graph_reasons or {}
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
                graph_reasons=graph_reasons.get(chunk_id, []),
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

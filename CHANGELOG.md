# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-13

### Added

- `ragless.graph`: graph-based retrieval built from structure already in
  your files -- no LLM involved in building or walking it.
- Edge builders (pluggable, individually weighted/enabled): `LinkEdges`
  (markdown/reference/wiki links, bare relative-path mentions, heading
  anchors), `ImportEdges` (Python via `ast`, JS/TS via regex), `SymbolEdges`
  (definition-to-usage, Python via `ast`, other languages via conservative
  regex), `StructureEdges` (sequential neighbors, heading hierarchy),
  `SharedTermEdges` (shared high-IDF terms, capped per node), and
  `SemanticKNNEdges` (embedding cosine similarity -- only added by default
  when the `semantic` extra's engine is configured).
- `search(..., graph=True, ...)` and `context(..., graph=True, ...)`: expand
  search hits across the graph with personalized PageRank (default) or
  bounded BFS, with `hops`, `max_expanded`, `edge_weights`, `min_score`, and
  `keep_seeds` (default `True` -- a seed is never outranked by expansion).
  Every expanded result carries its edge path back to a seed
  (`SearchResult.graph_reasons`, `Passage.reasons`).
- `Retriever.graph`: a lazily built, cached `ChunkGraph` with `neighbors()`,
  `related()`, `explain()`, `stats()`, and `to_json()`/`to_graphml()`/
  `to_networkx()` (networkx requires the new `graph` extra) export.
- The graph persists as a `graph.json` sidecar file in `cache_dir` and
  updates incrementally on file changes/additions/deletions/renames,
  matching a full rebuild exactly.
- `GraphExpand`: the graph-expansion step for the `ragless.context` pipeline,
  so expanded passages flow through window-expand/merge/dedup/diversify/
  compress/pack like any other passage.
- `edge_builders=` constructor argument on `Retriever` to fully replace the
  default edge-builder set.
- New `graph` extra (`networkx`, for `ChunkGraph.to_networkx()` only -- the
  rest of `ragless.graph` has no new required dependencies).
- New examples: `graph_retrieval.py`, `graph_export.py`.

## [0.2.0] - 2026-11-12

### Added

- `Retriever.context()`: turns search results into a `Context` fitted to a
  token budget, with citations, inclusion `reasons`, and matched-term
  highlight offsets.
- `ragless.context` pipeline: composable, swappable steps --
  `WindowExpand` (neighboring-line expansion), `MergeAdjacent` (combines
  overlapping/adjacent hits from the same file with correct line ranges),
  `Dedup` (fuzzy near-duplicate removal), `Diversify` (Maximal Marginal
  Relevance, using the semantic embedder when available or a lexical
  fallback otherwise), `Compress` (extractive sentence/line selection --
  never rewrites text), and `Pack` (greedy token-budget fitting that never
  exceeds the budget).
- `Context.to_text()/.to_markdown()/.to_xml()/.to_json()` output formats.
- `Retriever.prompt()`: renders a `Context` into a prompt string via a
  default template, a custom `str.format` template, or a custom callable.
  `ragless` still never calls an LLM anywhere.
- Pseudo-relevance feedback query expansion (`query_expansion=True`,
  off by default): RM3-style term expansion from the first pass's top
  results, re-searched and merged.
- Dependency-free token estimation by default; exact counts via
  `ragless.context.TiktokenCounter` under the new `tokens` extra
  (`pip install 'ragless[tokens]'`).
- New examples: `context_for_llm.py`, `prompt_templates.py`.

## [0.1.0] - 2026-11-12

### Added

- `Retriever`: library-first API indexing a directory or in-memory text.
- `LineChunker`: overlapping line-window chunking that preserves file path
  and 1-indexed start/end line numbers.
- `BM25Retriever`: Okapi BM25 keyword search (pure NumPy, no extra deps).
- `FuzzyRetriever`: fuzzy string matching via `rapidfuzz`.
- `SemanticRetriever`: local embedding search (`fastembed`, optional
  `semantic` extra) with NumPy cosine similarity.
- Reciprocal Rank Fusion (`reciprocal_rank_fusion`) to combine engine
  rankings, with configurable weights and `k`.
- `CrossEncoderReranker`: optional cross-encoder reranking of fused results.
- On-disk embedding cache (`EmbeddingCache`) as plain `.npy`/`.json` sidecar
  files, keyed by file mtime/size, so only changed files are re-embedded.
- Metadata and path filters on `search()`.
- Pluggable chunkers, retrievers, and embedders.

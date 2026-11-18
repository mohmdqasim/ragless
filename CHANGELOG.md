# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.2] - 2026-11-18

### Fixed

- `SemanticKNNEdges` re-embedded every chunk in the corpus in one call on
  *every* `r.graph` access, regardless of whether a `SemanticRetriever`
  already held those vectors in memory (warm `cache_dir`) or they were
  sitting in the on-disk embedding cache. On a 600-file / ~937-chunk corpus
  this meant ~160s and ~7.6GB peak memory on a warm restart, and embedding
  everything twice on a cold start. It now reuses an existing
  `SemanticRetriever`'s vectors (new `SemanticRetriever.vectors_by_chunk_id()`),
  falls back to the on-disk `EmbeddingCache` per source file, and only calls
  the embedder for chunks still missing after that -- in batches of
  `embed_batch_size` (default 32) to keep memory bounded. A warm restart
  with an unchanged corpus now embeds nothing; edges are unchanged.

## [0.3.1] - 2026-11-18

### Fixed

- `SymbolEdges` was O(definitions x chunks) -- it rescanned every chunk's
  full text for every definition. Now tokenizes each chunk once and builds
  a name -> chunks inverted index, so each definition is a dict lookup
  instead of a corpus rescan (~27s -> ~11ms on a 930-chunk/~1,300-definition
  corpus in testing). Edges are identical to the old implementation for
  plain-word identifiers (tested directly against a kept-for-comparison copy
  of the old algorithm); `$` in JS/TS identifiers is now also handled
  correctly in both the definition regex and the usage tokenizer.
- `FuzzyRetriever` ran a fuzzy-match scorer against every chunk's full text
  per query. Now defaults to `mode="vocabulary"`: fuzzy-matches query terms
  against the (much smaller) corpus vocabulary, then scores chunks
  BM25-style by the matched terms -- faster and less noisy (a typo no
  longer pulls in chunks that merely share a long substring). The old
  behavior is still available via `mode="full_text"`.
- `search(graph=True)`/`context(graph=True)` with the default
  `keep_seeds=True` never surfaced expanded results whenever there were
  already `top_k` or more seeds, making graph expansion invisible at normal
  result sizes. `keep_seeds=True` now only protects the top
  `max(1, top_k // 2)` seeds from being outranked by expansion (tunable via
  the new `protect_top_seeds=` argument); the rest of the result budget is
  open for expansion to compete for.

### Added

- A benchmark test asserting a warm-cache `Retriever` start plus `.graph`
  access stays fast (well under a second in practice) and median hybrid
  query latency stays low on a ~1,000-chunk corpus.

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

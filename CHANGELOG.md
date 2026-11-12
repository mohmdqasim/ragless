# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

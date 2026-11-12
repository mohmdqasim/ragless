# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

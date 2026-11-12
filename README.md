# ragless

File-based retrieval for RAG, with no vector database.

`ragless` indexes a folder of text files (or in-memory strings) and lets you
search them with BM25 keyword search, fuzzy string matching, and optional
local semantic embeddings -- combined with Reciprocal Rank Fusion. Everything
lives on disk as plain files (or in memory): no Postgres/pgvector, no
Pinecone/Weaviate/Chroma server, nothing to run.

```python
from ragless import Retriever

r = Retriever("./docs")
for hit in r.search("how do I configure retries", mode="hybrid"):
    print(f"{hit.path}:{hit.start_line}-{hit.end_line}  score={hit.score:.3f}")
    print(hit.text)
```

## Why

Most "RAG" tutorials reach for a vector database before checking whether they
need one. For a folder of a few thousand files, plain keyword search (BM25),
fuzzy matching, and a small local embedding model get you most of the way
there, with:

- No server to run, no API keys, no network calls (semantic search runs a
  small model locally via [fastembed](https://github.com/qdrant/fastembed)).
- No ingestion pipeline: point it at a directory and it chunks and indexes on
  construction.
- Caching as plain sidecar files (`.npy` + `.json`) next to your index, not a
  database you have to back up and migrate.
- Every result keeps the file path and line range it came from.

| | ragless | vector DB (Pinecone/Chroma/pgvector) |
|---|---|---|
| Infra | none | a running database/service |
| Setup | `Retriever("./docs")` | provision, configure, ingest |
| Keyword search | built in (BM25) | usually bolted on separately |
| Fuzzy/typo tolerance | built in | usually absent |
| Good for | a few thousand files, local tools, CLIs, CI | large, constantly-changing, multi-tenant corpora |
| Cold start | instant | needs the service up and reachable |

If you need to search tens of millions of chunks with concurrent writers
across a fleet of machines, use a vector database. If you need to search the
docs folder sitting next to your code, `ragless` is simpler.

## Install

```bash
pip install ragless                # BM25 + fuzzy (no heavy deps)
pip install 'ragless[semantic]'    # + local embeddings and cross-encoder rerank
```

## Quickstart

```python
from ragless import Retriever

# Index a directory of files.
r = Retriever("./docs")
results = r.search("installation steps", mode="hybrid", top_k=5)

# Or index in-memory text directly -- no files required.
r = Retriever(
    [
        "ragless indexes folders of text files for retrieval.",
        "BM25, fuzzy matching, and semantic search are combined with RRF.",
    ]
)
results = r.search("how are results combined?")

for hit in results:
    print(hit.path, hit.start_line, hit.end_line, hit.score)
    print(hit.text)
```

### Search modes

```python
r.search("query", mode="hybrid")  # fuse every configured engine (default)
r.search("query", mode="bm25")  # keyword search only
r.search("query", mode="fuzzy")  # fuzzy string matching only
r.search("query", mode="semantic")  # embeddings only (requires the extra)
```

### Semantic search and caching

```python
from ragless import Retriever

r = Retriever(
    "./docs",
    engines=["bm25", "fuzzy", "semantic"],
    cache_dir="./.ragless_cache",  # embeddings cached here as .npy/.json
)
r.search("query", mode="hybrid")
```

Embeddings are cached per source file, keyed by the file's mtime/size and the
embedder's name. Editing one file only re-embeds that file's chunks on the
next run; untouched files are loaded straight from the cache.

### Cross-encoder rerank

```python
r.search("query", mode="hybrid", rerank=True)
```

Reranking takes the top fused candidates and rescoring them with a small
local cross-encoder (also via `fastembed`, under the `semantic` extra) for
higher precision at the top of the list.

### Filters

```python
# Only chunks whose metadata says category="guide"
r.search("query", filters={"category": "guide"})

# Only chunks from markdown files
r.search("query", path_filter="*.md")

# Per-document metadata, assigned when building the index
r = Retriever(
    "./docs",
    metadata=lambda path: {"category": "guide"} if path and "guide" in path else {},
)
```

### Pluggable components

Chunkers, retrievers, and embedders are all swappable:

```python
from ragless import Retriever, Chunker, BaseRetriever
from ragless.chunking import LineChunker

r = Retriever("./docs", chunker=LineChunker(lines_per_chunk=60, overlap=10))

# Bring your own retriever/embedder implementation:
r = Retriever("./docs", engines={"bm25": MyRetriever(), "fuzzy": FuzzyRetriever()})
```

Implement `ragless.chunking.Chunker`, `ragless.retrievers.base.BaseRetriever`,
or `ragless.embedders.base.Embedder` to plug in your own.

## How it works

1. **Load** -- read files from a directory (or accept raw in-memory strings).
   Binary files and undecodable text are skipped automatically.
2. **Chunk** -- split each document into overlapping line-based windows,
   keeping the file path and 1-indexed `start_line`/`end_line` for every
   chunk (`ragless.chunking.LineChunker`, pluggable).
3. **Index** -- each configured engine builds its own index over the chunks:
   - `BM25Retriever`: classic Okapi BM25 over tokenized text (NumPy).
   - `FuzzyRetriever`: `rapidfuzz` string similarity.
   - `SemanticRetriever`: dense embeddings (`fastembed`) with cosine
     similarity (NumPy), cached to disk as `.npy`/`.json` sidecar files.
4. **Search** -- query every configured engine, fuse their rankings with
   [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf),
   apply metadata/path filters, and optionally rerank the top candidates with
   a cross-encoder.

## Status

0.1.0 -- the API may still shift. Feedback and issues welcome.

## License

MIT

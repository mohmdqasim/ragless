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

## Context for your LLM

`ragless` handles everything up to the LLM call: `r.context()` turns a query
into ready-to-use, budget-fitted context. There's no LLM inside `ragless` --
it never makes a network call or imports an LLM client; it just hands you the
material to send to whichever model you call yourself.

```python
from ragless import Retriever

r = Retriever("./docs")
ctx = r.context("how do I configure retries", max_tokens=2000)

print(ctx.token_count)  # never exceeds max_tokens
for citation in ctx.citations:
    print(citation)  # [1] docs/retries.md:12-34
```

Under the hood, `context()` runs a pipeline of small, swappable steps:
expand hits with neighboring lines, merge adjacent hits from the same file,
drop near-duplicates, diversify with MMR, optionally compress to the most
relevant sentences/lines, then pack whole passages into the token budget
(never truncating mid-passage). Every passage keeps a `reasons` list (e.g.
`["direct match"]`, `["direct match", "neighbor window"]`) explaining why it
was included.

### Output formats

```python
ctx.to_text()  # plain text with trailing "[n] path:start-end" citations
ctx.to_markdown()  # fenced code blocks for code paths, blockquotes for prose
ctx.to_xml()  # <context><passage index="1" ref="...">...</passage></context>
ctx.to_json()  # full structure, including matched-term highlight offsets
ctx.to_text(mark=True)  # wraps matched query terms, e.g. with >>term<<
```

### Prompts -- still no LLM call

```python
prompt = r.prompt(
    "how do I configure retries",
    max_tokens=2000,
    template="Context:\n{context}\n\nQuestion: {query}\nAnswer:",  # optional
)

# Send it to any client yourself:
# response = my_llm_client.complete(prompt)
```

### Tuning the pipeline

```python
r.context(
    "query",
    max_tokens=2000,
    window=3,  # lines of neighboring context to pull in
    dedup_threshold=90,  # fuzzy-ratio cutoff for near-duplicates
    diversify=True,  # reorder by Maximal Marginal Relevance
    compress=True,  # keep only the most query-relevant sentences/lines
    query_expansion=True,  # pseudo-relevance feedback (RM3-style), off by default
)
```

Exact token counts (instead of the dependency-free default estimate) are
available via `pip install 'ragless[tokens]'`:

```python
from ragless import Retriever
from ragless.context import TiktokenCounter

r = Retriever("./docs")
r.context("query", max_tokens=2000, token_counter=TiktokenCounter())
```

### Custom pipeline steps

```python
from ragless.context import ContextStep, Pack


class DropShortPassages(ContextStep):
    def run(self, passages, state):
        return [p for p in passages if len(p.text) > 40]


r.context("query", max_tokens=2000, pipeline=[DropShortPassages(), Pack()])
```

A pipeline step only needs `run(passages, state) -> passages`, so a future
graph-expansion step can plug in the same way: add candidate passages with
their own `reasons` entry, without touching merge/dedup/pack.

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
5. **Context** -- `r.context()` runs search results through a pipeline (expand
   -> merge -> dedup -> diversify -> compress -> pack) and returns a
   `Context` fitted to a token budget, with citations and highlight offsets.
   `r.prompt()` renders that into a prompt string. No LLM call happens
   anywhere in `ragless`.

## Roadmap

- **Graph-based retrieval** (planned next): expand candidates by following
  references between chunks (e.g. imports, links, call graphs) as another
  pluggable `context` pipeline step, so related-but-not-directly-matched
  passages can be pulled in with their own `reasons` entry.

## Status

0.2.0 -- the API may still shift. Feedback and issues welcome.

## License

MIT

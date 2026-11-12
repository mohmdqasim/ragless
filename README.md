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

## Graph retrieval

Search hits become seeds, and a graph built from structure *already in your
files* -- links, imports, symbol definitions/usages, heading hierarchy,
shared rare terms, and (optionally) embedding similarity -- pulls in related
chunks the query never mentioned. No LLM is involved in building or walking
this graph; it's the opposite of LLM-based GraphRAG, which uses a model to
extract entities/relations. Here the "relations" are just the links, imports,
and symbols that were already sitting in your files.

```python
from ragless import Retriever

r = Retriever("./docs")

# follows links/imports/symbols/etc. up to 2 hops from the search hits
for hit in r.search("how does login retry work", graph=True):
    print(hit.path, hit.score, hit.graph_reasons)  # e.g. ["auth.md links to retry.md"]

# graph expansion is just another step in the v0.2 context pipeline
ctx = r.context("how does login retry work", max_tokens=2000, graph=True)
```

Off by default; turn it on with `graph=True` on `search()` or `context()`.
Tune it with `hops`, `max_expanded`, `graph_mode` (`"pagerank"` or `"bfs"`),
`edge_weights` (reweight edge types at query time, e.g. `{"link": 2.0}`,
without rebuilding the graph), `min_score`, and `keep_seeds` (default
`True`: a seed is never outranked by something the graph pulled in).

### Inspecting the graph directly

```python
r.graph.stats()  # {"nodes": ..., "edges": ..., "edges_by_type": {...}}
r.graph.related("docs/auth.md", k=5)  # other files most connected to this one
r.graph.explain(some_chunk_id)  # every edge touching this chunk, with its reason
r.graph.neighbors(some_chunk_id, edge_types=["import"])

r.graph.to_json()
r.graph.to_graphml()  # plain XML, no extra dependency
r.graph.to_networkx()  # requires `pip install ragless[graph]`
```

### Custom edge builders

```python
from ragless import Retriever
from ragless.graph import LinkEdges, ImportEdges, EdgeBuilder


class MyEdges(EdgeBuilder):
    name = "my_edges"

    def build(self, ctx): ...  # see ragless.graph.builders for the BuildContext API


r = Retriever("./docs", edge_builders=[LinkEdges(weight=2.0), ImportEdges(), MyEdges()])
```

The graph persists as a plain `graph.json` sidecar file in `cache_dir` (the
same directory used for the embedding cache) and updates incrementally when
files change, add, delete, or rename -- only the affected files are
re-scanned; the result matches a full rebuild exactly.

### Does it actually help?

On a small multi-hop benchmark (queries whose answer is only in a file
*linked from* the directly-matched one -- a support doc pointing at a policy
doc, a client module importing the function that does the real work) with
3 such queries plus 1 distractor:

| | recall@3 | precision@3 |
|---|---|---|
| hybrid search alone | 0.50 | 0.167 |
| with `graph=True` | 1.00 | 0.333 |

The honest case where it doesn't help: a query whose answer is in an
isolated file with no links, imports, or shared distinctive terms to anything
else -- graph expansion has nothing to expand into, so results are identical
with or without it. If your corpus is mostly standalone files, don't expect
much lift.

On a synthetic ~5,000-chunk corpus (700 linked markdown files): building the
graph took ~4.9s, and an average `graph=True` query (PageRank, 2 hops) added
~25ms over plain hybrid search. Reproduce with
`pytest --run-slow -m slow tests/test_graph_benchmark.py -s`.

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
6. **Graph** (optional) -- `r.graph` builds a typed, weighted graph over
   chunks from structure already in the files (links, imports, symbols,
   headings, shared rare terms); `graph=True` on `search()`/`context()`
   expands search hits across it (personalized PageRank or bounded BFS) to
   pull in related chunks the query never mentioned.

## Roadmap

- Query-time re-ranking that blends graph centrality with relevance scores
  more flexibly than the current fixed combination formula.
- Incremental graph updates for the two corpus-wide edge builders (symbols,
  shared terms) -- currently always fully recomputed on any change; see
  `ragless.graph.ChunkGraph` for the local/global builder split.

## Status

0.3.0 -- the API may still shift. Feedback and issues welcome.

## License

MIT

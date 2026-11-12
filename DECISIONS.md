# v0.2 design decisions

Working notes for the `ragless.context` feature (the "A" in RAG). Not shipped
to PyPI as documentation; this is an internal log of choices made while
building autonomously, so the reasoning doesn't have to be reconstructed later.

## Module layout

`ragless/context/`:
- `tokens.py` -- `estimate_tokens` (chars/4 heuristic, no deps) and
  `TiktokenCounter` (lazy-imports `tiktoken`, requires the new `tokens` extra).
- `passage.py` -- `Passage`, `Citation`, `Context` (with `to_text/to_markdown/
  to_xml/to_json`) and the default prompt template.
- `pipeline.py` -- `ContextStep` base class and the six built-in steps
  (`WindowExpand`, `MergeAdjacent`, `Dedup`, `Diversify`, `Compress`, `Pack`),
  plus `build_default_pipeline()`.
- `query_expansion.py` -- RM3-style pseudo-relevance feedback, off by default.
- `__init__.py` -- `build_context()` orchestration + public re-exports.

`Retriever.context()` and `Retriever.prompt()` live in `core.py` (lazy-import
`ragless.context` to avoid a cycle) so the public entry point stays
`r.context(...)` / `r.prompt(...)`, matching the existing `r.search(...)`
style instead of a separate builder object the user has to construct.

## Why a step-based pipeline instead of one big function

Each step is a small class implementing `run(passages, state) -> passages`.
`state` (`PipelineState`) carries the query, token budget/counter, and a
document-text/embedder accessor so steps don't each need their own
constructor wiring for things that are per-call, not per-step. Steps only
take constructor args for their own tunables (window size, dedup threshold,
MMR lambda, ...). This lets `pipeline=[...]` fully replace the default list,
and a future graph-expansion step (v0.3) only needs to append/prepend another
step that adds candidate passages with its own `reasons` entry -- it doesn't
need to know about merge/dedup/pack internals.

## Passage identity and merging across chunk boundaries

Merging adjacent/overlapping hits needs the *original* document text, not
just concatenated chunk text, because `LineChunker` chunks overlap by
default -- naively joining two chunks would duplicate the overlapping lines.
So `Retriever` now also keeps `self._doc_text: dict[str, str]` (doc key ->
full original text) and `self._chunk_doc_id: dict[str, str]` (chunk id -> doc
key), built during the existing chunking loop in `__init__`. Both are
private/internal; chunking itself and the public `Chunk` dataclass are
unchanged, so this doesn't touch the frozen `Chunk` API or any existing test
that constructs `Chunk(...)` directly. Doc keys are the same `path` or
`f"mem:{i}"` prefixes already used for chunk ids, so merged/expanded passages
from in-memory text get a citation id like `mem:0` instead of a path.

`MergeAdjacent` and `WindowExpand` re-slice from `self._doc_text[doc_id]`
by line range rather than editing chunk text, so a merged passage's text is
always an exact line-range slice of the source, and `start_line`/`end_line`
stay accurate for citations.

## Diversity (MMR)

Uses the retriever's own semantic embedder when `engines` includes
`"semantic"` (found via `retriever._retrievers.get("semantic").embedder`);
otherwise falls back to a lexical similarity matrix from
`rapidfuzz.fuzz.token_sort_ratio`. Either way `Diversify` only needs a
`similarity(i, j) -> float in [0, 1]` function, so the embedder dependency is
optional and never required at import time.

## Extractive compression

Sentence-level for prose, line/block-level for recognized code file
extensions (`_CODE_EXTS`). Scoring is lexical token overlap with the query
(no embeddings required, since `compress` has to work without the `semantic`
extra too). Compression only *selects* a subsequence of the original
sentences/lines -- it never rewrites text -- and joins non-contiguous kept
spans with a literal `"\n[...]\n"` marker so the output is still clearly
attributable to the source. Tests assert every span between markers is an
exact substring of the original passage text.

## Token budget (`Pack`)

Greedy: steps through passages in their current order and keeps a passage
only if it still fits the remaining budget, so the total is a hard max, not a
best-effort trim -- "budget smaller than one passage" yields an empty
`Context` rather than a truncated one, since clipping mid-passage would also
clip mid-citation. Default token counter is a dependency-free chars/4
estimate; `TiktokenCounter` (needs `pip install ragless[tokens]`) or a fully
custom `Callable[[str], int]` can replace it via `token_counter=`.

## Query expansion (RM3-style)

Off by default (`query_expansion=False` on `context()`). When enabled: run
the initial search, pull term frequencies from the top few hits' text
(excluding a small stopword list and terms already in the query), append the
top N new terms to the query, re-search, and merge candidate pools by chunk
id (keep the max score). If no new terms clear the frequency threshold
(small corpus, all terms already in the query, etc.) it falls back to the
unexpanded candidates untouched rather than erroring.

## Output formats and prompts

`Context.to_text/to_markdown/to_xml/to_json()` are plain string builders, no
templating engine. `r.prompt(query, template=..., **context_kwargs)` accepts
either a `str.format`-style template with `{query}`/`{context}` placeholders
(context rendered via `context_format=` before being formatted in) or a
callable `(query, Context) -> str` for full control. `ragless` itself never
imports an LLM client or makes a network call anywhere in this module.

## No new required dependencies

`ragless.context` only imports `numpy` and `rapidfuzz` (already required by
v0.1) plus the stdlib. `tiktoken` lives behind the new `tokens` extra and is
only imported inside `TiktokenCounter.__init__`.

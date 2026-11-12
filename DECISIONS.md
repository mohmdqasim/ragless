# Autonomous build decisions

Decisions made while building `ragless` without the user available to ask.

## Packaging / tooling

- **Build backend**: hatchling with a `src/` layout. Simple, no `setup.py`,
  plays well with editable installs and PyPI trusted publishing.
- **Python support**: `>=3.10`. Chosen as a modern-but-not-bleeding-edge
  floor; CI covers 3.10-3.13 on Ubuntu and macOS.
- **GitHub owner/repo**: created under the authenticated `gh` account
  `mohmdqasim`, repo name `ragless` (public). `pyproject.toml` URLs point
  at `https://github.com/mohmdqasim/ragless`.
- **Linting**: `ruff` (check + format), configured in `pyproject.toml`
  (`[tool.ruff]`). CI fails on any lint or formatting diff.
- **Coverage gate**: `--cov-fail-under=85` in CI, matching the 85%+ target.
  Actual achieved coverage on the fast suite is ~90%.

## Dependencies

- **BM25**: implemented from scratch (`ragless/retrievers/bm25.py`) using
  only NumPy, rather than pulling in `rank-bm25` or a search engine. Keeps
  the core dependency surface to just `numpy` + `rapidfuzz`.
- **Fuzzy**: `rapidfuzz` (explicitly requested in the spec), used via
  `rapidfuzz.process.extract`.
- **Semantic embeddings**: `fastembed` (ONNX runtime-based), not
  `sentence-transformers`/`torch`. Much lighter install for an optional
  extra, still fully local with no API keys. Default model:
  `BAAI/bge-small-en-v1.5` (verified available in fastembed's supported
  model list at build time).
- **Cross-encoder rerank**: also via `fastembed` (`fastembed.rerank.
  cross_encoder.TextCrossEncoder`), under the same `semantic` extra rather
  than a separate `rerank` extra, since it's the same dependency. Default
  model: `Xenova/ms-marco-MiniLM-L-6-v2`.
- Both default model names were confirmed against the installed fastembed
  version's `list_supported_models()` before being hardcoded.

## API shape

- `Retriever(source, *, texts=None, chunker=None, engines=None,
  embedder=None, cache_dir=None, include=None, exclude=None, metadata=None,
  encoding="utf-8", rrf_k=60, weights=None)`.
  - `source` accepts a directory path, a single file path, or a sequence of
    raw strings (in-memory documents) directly as the first positional arg,
    per the spec's "should also accept in-memory text". `texts=` is an
    additional kwarg so in-memory notes can be mixed in alongside a
    directory source.
  - `engines` defaults to `("bm25", "fuzzy")` only -- semantic is opt-in via
    `engines=[..., "semantic"]`, keeping the core install light as required.
    It also accepts a `Mapping[str, BaseRetriever]` for full pluggability
    (bring your own retriever implementation under any engine name).
  - `metadata` can be a constant dict or a `path -> dict` callable, so
    per-file metadata tagging doesn't require a separate data structure.
- **Filters** are two separate mechanisms, matching the spec's "metadata and
  path filters": `filters={"key": value_or_predicate}` against chunk
  metadata, and `path_filter="*glob*"` (or a list of globs) against the
  chunk's file path. In-memory chunks (`path is None`) never match a
  `path_filter`.
- **Fusion**: Reciprocal Rank Fusion with `k=60` (the constant from the
  original Cormack et al. RRF paper), configurable per-`Retriever` via
  `rrf_k=` and per-engine `weights=`.
- **Caching**: one `.npy` + `.json` sidecar pair per *source file* (not per
  chunk), keyed by `sha256(path :: embedder_name)[:24]` to keep filenames
  short and collision-resistant, with the original path stored inside the
  JSON for debuggability. Invalidated on `(mtime, size)` signature change,
  embedder name/dim change, or chunk id list change (covers chunker/config
  changes too, not just file edits). In-memory text is never cached (no
  stable file identity to invalidate against).
- Binary files and files that fail to decode under the given `encoding` are
  silently skipped rather than raising, matching "works on folders" robustly
  without per-file configuration. (Detected via a null-byte check in the
  first 8KB, then a decode attempt.)

## Testing

- `tests/conftest.py` provides a dependency-free `FakeEmbedder` (hashed
  bag-of-words) so semantic-retriever logic, caching, and fusion can all be
  tested without downloading a real model.
- Real-model tests (`tests/test_semantic_real_model.py`) are marked
  `@pytest.mark.slow` and additionally gated by a `--run-slow` CLI flag
  (added in `conftest.py`); they're skipped by default and run in a
  dedicated CI job (`test-semantic-slow`) that installs the `semantic`
  extra. The fast test suite never needs network access or fastembed
  installed (verified by running it in a venv with no semantic extra).
- "Missing optional extra" behavior is tested by monkeypatching
  `sys.modules["fastembed"] = None` rather than requiring a real
  extra-less environment for every test run -- `FastEmbedEmbedder` and
  `CrossEncoderReranker` both import `fastembed` lazily inside `__init__`,
  so this reliably triggers the same `ImportError` path.

## CI / publishing

- `ci.yml`: `lint` (ruff) -> `test` (matrix: ubuntu/macos x 3.10-3.13,
  `--cov-fail-under=85`) + `test-semantic-slow` (ubuntu/3.12 only, installs
  `semantic` extra, runs `--run-slow -m slow`) -> `build` (sdist/wheel +
  wheel-install smoke test).
- `publish.yml`: triggers on `v*` tags; re-runs the test matrix and a build
  job *inside this workflow* (a separate workflow run can't `needs:` another
  workflow's jobs), then `publish` (environment `pypi`, `permissions:
  id-token: write`) uses `pypa/gh-action-pypi-publish` with OIDC trusted
  publishing -- no API token stored in the repo.
- **PyPI trusted publisher registration is a manual, credential-gated step**
  that this session cannot perform (it requires logging into pypi.org as
  the account that will own the `ragless` project). See `PROGRESS.md` for
  exactly what to do.

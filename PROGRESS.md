# v0.2 progress

## Status: done

- [x] Read existing v0.1 code/tests/README/CI, confirm no prior v0.2 work.
- [x] Write DECISIONS.md with the v0.2 architecture.
- [x] Implement `ragless.context` package (tokens, passage/context, pipeline
      steps, query expansion, orchestration).
- [x] Wire `Retriever.context()` / `Retriever.prompt()` into `core.py`, add
      the internal `_doc_text` / `_chunk_doc_id` bookkeeping.
- [x] Update `ragless/__init__.py` exports + bump version to 0.2.0.
- [x] Tests for every pipeline step + integration tests + edge cases.
- [x] Run v0.1 tests (must still pass), ruff, coverage >= 85%.
- [x] Build wheel, `twine check`, install + smoke test in a fresh venv.
- [x] README "Context for your LLM" section, CHANGELOG, new examples,
      Roadmap section.
- [x] Commit in small logical commits, push, green CI.
- [x] Tag v0.2.0, push tag, watch publish workflow.
- [x] Verify `pip install ragless==0.2.0` + smoke test `r.context(...)`.
- [x] GitHub release for v0.2.0.

## What was built

`ragless.context`: a composable pipeline (`expand -> merge -> dedup ->
diversify -> compress -> pack`) that turns `Retriever.search()` results into
a token-budgeted `Context` -- citations, inclusion `reasons`, matched-term
highlight offsets, and `to_text()/to_markdown()/to_xml()/to_json()` output.
`Retriever.prompt()` renders that into a prompt string (default template,
custom string template, or custom callable). No LLM client or network call
anywhere in the library. Optional pseudo-relevance feedback query expansion
(`query_expansion=True`, off by default). New `tokens` extra for exact
token counts via `tiktoken` (`TiktokenCounter`); the default counter is a
dependency-free character-based estimate, so the core install stays light.
Every pipeline step is a small class (`run(passages, state) -> passages`),
so a future graph-expansion step (v0.3) can plug in the same way, and
`pipeline=[...]` lets a caller fully replace the default steps.

Full design reasoning (module layout, merge/expand correctness, MMR,
compression, token packing, query expansion) is in `DECISIONS.md`.

## Tests and coverage

134 fast tests + 4 slow tests (real embedding model, real tiktoken
encoding, slow example), all passing. Coverage 92.7% (gate is 85%). `ruff
check` and `ruff format --check` both clean. All v0.1 tests pass unchanged
-- no public API was broken.

## CI

One CI run hit a transient GitHub Actions macOS-runner capacity issue
(`macos-latest, 3.11` queued and was cancelled twice, ~15 min each time;
the run's own annotation confirmed "capacity constraints, jobs targeting
macOS arm64 runners may experience longer queue times"). Reran that single
job and it completed in 17s. Full CI (lint, 8-way test matrix, semantic
slow tests, build) is green:
https://github.com/mohmdqasim/ragless/actions/runs/43253313730

## Release

- Tag: `v0.2.0`
- Publish workflow (OIDC trusted publishing, no stored token): green --
  https://github.com/mohmdqasim/ragless/actions/runs/43274824455
- PyPI: https://pypi.org/project/ragless/0.2.0/
- GitHub release: https://github.com/mohmdqasim/ragless/releases/tag/v0.2.0
- Verified `pip install ragless==0.2.0` in a fresh venv and ran a live
  `r.context(...)` / `r.prompt(...)` smoke test against the published
  package.

## Notes

`gh`'s active account had silently switched back to the wrong org-like
account (`qasim-ensi-ai`, no push access) before the push step, same as
during the v0.1 session. Fixed with `gh auth switch --hostname github.com
--user mohmdqasim` before pushing; no repo/account changes were otherwise
needed since v0.1 already lived at the correct `mohmdqasim/ragless`.

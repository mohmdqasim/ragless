# Progress

Autonomous build log for `ragless`. Updated as work proceeds.

## Status: done -- published to PyPI as ragless 0.1.0

### Done

- [x] Full package implemented under `src/ragless/` (chunking, BM25, fuzzy,
      semantic, fusion, rerank, cache, core `Retriever`), pluggable
      throughout, with type hints and docstrings.
- [x] Test suite: 61 fast tests (chunker, each retriever, fusion, cache
      invalidation, filters, directory/in-memory loading, binary/empty/
      unicode/large-file edge cases, missing-extra error messages, custom
      reranker) + 2 slow tests gated behind `--run-slow`/`-m slow`.
- [x] Fast suite passes in a venv **without** the `semantic` extra installed
      (confirms the core install is genuinely light).
- [x] Slow suite passes with the real `fastembed` model
      (`BAAI/bge-small-en-v1.5` + `Xenova/ms-marco-MiniLM-L-6-v2`).
- [x] `ruff check .` and `ruff format --check .` both clean.
- [x] Coverage: 90% (gate set at 85% in CI).
- [x] `python -m build` produces sdist + wheel; installed the wheel into a
      fresh venv and ran an import + search smoke test successfully.
- [x] README (quickstart, API examples, how it works, comparison table),
      CHANGELOG.md, `examples/` (4 runnable scripts, all manually run and
      verified, including the semantic+rerank one).
- [x] `.github/workflows/ci.yml` (lint, test matrix ubuntu/macos x
      3.10-3.13, slow-model job, build+smoke-test job).
- [x] `.github/workflows/publish.yml` (triggers on `v*` tags, reruns tests,
      builds, then publishes via OIDC trusted publishing to the `pypi`
      environment).

- [x] `git init` + initial commit, history kept to clean, correctly-authored
      commits (`Qasim Arshad <mohmdqasim0@gmail.com>`), no AI/trailer
      mentions anywhere.
- [x] `twine check dist/*` passes (sdist + wheel); PyPI page will render the
      full README as `text/markdown` with correct author/maintainer and
      project URLs (Homepage/Repository/Issues/Changelog).
- [x] Public GitHub repo created at https://github.com/mohmdqasim/ragless
      (the repo was first created under the wrong, auto-selected `gh`
      account and was recreated here after the user logged `gh` into their
      personal account; the old repo was deleted by the user).
- [x] `pypi` GitHub Environment created on the repo via `gh api`.
- [x] CI green on `main`: lint, test matrix (ubuntu/macos x 3.10-3.13),
      slow-model job, and build+smoke-test job all passed
      (https://github.com/mohmdqasim/ragless/actions).

- [x] User set up the PyPI trusted publisher (pending publisher for
      owner=`mohmdqasim`, repo=`ragless`, workflow=`publish.yml`,
      environment=`pypi`) before the tag was pushed.
- [x] Tagged `v0.1.0`, pushed the tag (clean tagger identity, no AI
      mentions). `publish.yml` ran: test matrix -> build -> publish, all
      green, via OIDC trusted publishing (no token ever stored).
- [x] Verified with `pip install ragless` in a fresh venv: imports, searches
      in-memory text, returns the expected result.

## Links

- GitHub: https://github.com/mohmdqasim/ragless
- PyPI: https://pypi.org/project/ragless/
- CI: https://github.com/mohmdqasim/ragless/actions

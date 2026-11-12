# Progress

Autonomous build log for `ragless`. Updated as work proceeds.

## Status: build complete, CI green on GitHub, publishing blocked on a manual PyPI step

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

### Next

- [ ] Tag `v0.1.0`, push the tag, watch the publish workflow run.
- [ ] Verify with `pip install ragless` in a fresh venv + a quick search.

## Blocking item: PyPI trusted publisher registration

**This cannot be completed autonomously** -- it requires logging into
pypi.org with human credentials, which this session does not have and
should not attempt to obtain.

`publish.yml` is wired for PyPI's OIDC "trusted publishing": no API token is
stored anywhere, but PyPI must be told in advance which GitHub repo/workflow
is allowed to publish the `ragless` project. For a project that has never
been published before, this is done once via PyPI's web UI as a **pending
publisher**, which reserves the name and links it to this workflow ahead of
the first publish.

### What you need to do (takes ~1 minute)

1. Log into https://pypi.org (create an account first if you don't have
   one).
2. Go to https://pypi.org/manage/account/publishing/
3. Under "Add a new pending publisher", fill in:
   - **PyPI project name**: `ragless`
   - **Owner**: `mohmdqasim`
   - **Repository name**: `ragless`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
4. Submit. That's it -- no token to copy anywhere.

Once that's set up, the next tag push (or a re-run of a failed `publish.yml`
run) will publish successfully. If this step hasn't been done yet when this
session tags `v0.1.0` and pushes it, the `publish` job in the workflow run
will fail at the PyPI upload step with an authentication error -- that is
expected and not a bug in the workflow; it just means the pending publisher
above still needs to be registered, after which re-running that same
workflow run (via the GitHub Actions UI, or pushing a new tag like `v0.1.1`)
will succeed.

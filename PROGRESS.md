# v0.2 progress

## Status: in progress

- [x] Read existing v0.1 code/tests/README/CI, confirm no prior v0.2 work.
- [x] Write DECISIONS.md with the v0.2 architecture.
- [ ] Implement `ragless.context` package (tokens, passage/context, pipeline
      steps, query expansion, orchestration).
- [ ] Wire `Retriever.context()` / `Retriever.prompt()` into `core.py`, add
      the internal `_doc_text` / `_chunk_doc_id` bookkeeping.
- [ ] Update `ragless/__init__.py` exports + bump version to 0.2.0.
- [ ] Tests for every pipeline step + integration tests + edge cases.
- [ ] Run v0.1 tests (must still pass), ruff, coverage >= 85%.
- [ ] Build wheel, `twine check`, install + smoke test in a fresh venv.
- [ ] README "Context for your LLM" section, CHANGELOG, new examples,
      Roadmap section.
- [ ] Commit in small logical commits, push, green CI.
- [ ] Tag v0.2.0, push tag, watch publish workflow.
- [ ] Verify `pip install ragless==0.2.0` + smoke test `r.context(...)`.
- [ ] GitHub release for v0.2.0.

Will fill in the final summary (what was built, test/coverage numbers,
release links) at the end.

from __future__ import annotations

from dataclasses import replace

from ragless import Context, LineChunker, Retriever
from ragless.context.pipeline import ContextStep, Pack


def test_context_respects_token_budget(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context("cached embeddings installation search", max_tokens=200)
    assert isinstance(ctx, Context)
    assert ctx.token_count <= 200
    assert sum(len(p.text) for p in ctx.passages) > 0


def test_context_citations_match_passages_in_order(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context("installation", max_tokens=2000)
    assert [c.ref for c in ctx.citations] == [p.ref for p in ctx.passages]
    assert [c.index for c in ctx.citations] == list(range(1, len(ctx.passages) + 1))


def test_context_output_formats_all_produce_strings(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context("installation", max_tokens=2000)
    assert isinstance(ctx.to_text(), str)
    assert isinstance(ctx.to_markdown(), str)
    assert isinstance(ctx.to_xml(), str)
    assert isinstance(ctx.to_json(), str)


def test_context_merged_passages_have_exact_line_ranges(tmp_path):
    lines = [f"line{i} filler content" for i in range(1, 31)]
    lines[9] = "line10 mentions the refund policy explicitly"
    lines[19] = "line20 also discusses the refund policy in detail"
    content = "\n".join(lines) + "\n"
    (tmp_path / "doc.md").write_text(content)

    r = Retriever(str(tmp_path), chunker=LineChunker(lines_per_chunk=6, overlap=2))
    ctx = r.context("refund policy", max_tokens=5000, top_k=50)

    assert ctx.passages
    file_lines = content.splitlines()
    for p in ctx.passages:
        assert p.path is not None and p.path.endswith("doc.md")
        expected = "\n".join(file_lines[p.start_line - 1 : p.end_line])
        assert p.text == expected  # citation line range matches the source file exactly
        assert p.reasons  # every passage carries at least one inclusion reason


def test_context_no_results_is_empty():
    r = Retriever(["something about pandas"], engines=["bm25"])
    ctx = r.context("zzzzznonexistentqueryterm9999")
    assert ctx.passages == []
    assert ctx.citations == []
    assert ctx.token_count == 0
    assert ctx.to_text() == ""


def test_context_budget_smaller_than_one_passage(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context("installation", max_tokens=1)
    assert ctx.passages == []
    assert ctx.token_count == 0


def test_context_unicode_passage(tmp_path):
    (tmp_path / "unicode.txt").write_text(
        "café naïve 你好 résumé unique-marker-77\nsecond line here\n", encoding="utf-8"
    )
    r = Retriever(str(tmp_path))
    ctx = r.context("unique-marker-77", max_tokens=1000)
    assert ctx.passages
    assert "unique-marker-77" in ctx.passages[0].text


def test_context_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "content.txt").write_text("some content about onboarding guides")
    r = Retriever(str(tmp_path))
    ctx = r.context("onboarding guides", max_tokens=1000)
    assert ctx.passages
    assert all(p.path and p.path.endswith("content.txt") for p in ctx.passages)


def test_context_single_file_corpus(tmp_path):
    (tmp_path / "only.md").write_text("# Title\nsingle file corpus content about widgets\n")
    r = Retriever(str(tmp_path / "only.md"))
    ctx = r.context("widgets", max_tokens=1000)
    assert ctx.passages
    assert ctx.passages[0].path is not None and ctx.passages[0].path.endswith("only.md")


def test_context_in_memory_text_uses_mem_doc_ids():
    r = Retriever(["first note about pandas", "second note about rockets"])
    ctx = r.context("pandas", max_tokens=1000)
    assert ctx.passages
    assert ctx.passages[0].path is None
    assert ctx.passages[0].doc_id.startswith("mem:")
    assert ctx.citations[0].ref.startswith("mem:")


def test_context_compress_code_file_keeps_only_original_lines(tmp_path):
    code = "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "",
            "def unrelated():",
            "    pass",
            "",
            "def search(query):",
            "    return bm25.query(query)",
        ]
    )
    (tmp_path / "search.py").write_text(code)
    r = Retriever(str(tmp_path))
    ctx = r.context("search bm25 query", max_tokens=5000, compress=True, top_k=10)
    assert ctx.passages
    for line in ctx.passages[0].text.split("\n"):
        line = line.strip()
        if line and line != "[...]":
            assert line in code
    assert "```py" in ctx.to_markdown()


def test_context_with_semantic_extra_diversify(fake_embedder, sample_corpus):
    r = Retriever(str(sample_corpus), engines=["bm25", "fuzzy", "semantic"], embedder=fake_embedder)
    ctx = r.context(
        "cached embeddings search installation", max_tokens=2000, diversify=True, top_k=20
    )
    assert isinstance(ctx, Context)


def test_context_without_semantic_extra_diversify_lexical_fallback(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context("installation search caching", max_tokens=2000, diversify=True, top_k=20)
    assert isinstance(ctx, Context)


def test_query_expansion_runs_and_returns_results(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context("search", max_tokens=2000, query_expansion=True, top_k=20, diversify=False)
    assert isinstance(ctx, Context)
    assert ctx.passages


def test_query_expansion_no_useful_terms_falls_back_gracefully():
    r = Retriever(["the query is the query"])
    ctx = r.context("query", max_tokens=1000, query_expansion=True, top_k=10, diversify=False)
    assert isinstance(ctx, Context)
    assert ctx.passages


def test_custom_pipeline_replaces_default_steps(sample_corpus):
    class TagStep(ContextStep):
        def run(self, passages, state):
            return [replace(p, metadata={**p.metadata, "custom": True}) for p in passages]

    r = Retriever(str(sample_corpus))
    ctx = r.context("installation", max_tokens=2000, pipeline=[TagStep(), Pack()], top_k=10)
    assert ctx.passages
    assert all(p.metadata.get("custom") is True for p in ctx.passages)


def test_custom_token_counter_controls_budget(sample_corpus):
    r = Retriever(str(sample_corpus))
    ctx = r.context(
        "installation search caching",
        max_tokens=2,
        token_counter=lambda t: 1,
        top_k=20,
        diversify=False,
    )
    assert len(ctx.passages) <= 2


def test_prompt_default_template(sample_corpus):
    r = Retriever(str(sample_corpus))
    out = r.prompt("installation steps", max_tokens=1000)
    assert "Question: installation steps" in out
    assert "Context:" in out


def test_prompt_custom_string_template(sample_corpus):
    r = Retriever(str(sample_corpus))
    out = r.prompt("installation steps", template="Q:{query}|C:{context}", max_tokens=1000)
    assert out.startswith("Q:installation steps|C:")


def test_prompt_custom_callable_template(sample_corpus):
    r = Retriever(str(sample_corpus))
    out = r.prompt(
        "installation steps",
        template=lambda q, ctx: f"{q}::{len(ctx.passages)}",
        max_tokens=1000,
    )
    assert out.startswith("installation steps::")


def test_prompt_context_format_json(sample_corpus):
    r = Retriever(str(sample_corpus))
    out = r.prompt("installation steps", context_format="json", max_tokens=1000)
    assert '"query": "installation steps"' in out

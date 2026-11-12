from __future__ import annotations

import json

from ragless.context._lexical import compute_highlights
from ragless.context.passage import Context, Passage


def _ctx(passages):
    return Context(
        query="installation steps",
        passages=passages,
        token_count=sum(len(p.text) for p in passages),
    )


def test_compute_highlights_finds_word_boundary_matches():
    offsets = compute_highlights("Install ragless, then reinstall it", ["install"])
    assert offsets == [(0, 7)]  # "reinstall" does not match the whole-word "install"


def test_compute_highlights_no_terms_returns_empty():
    assert compute_highlights("some text", []) == []


def test_citations_are_numbered_in_order():
    passages = [
        Passage(text="a", doc_id="a.md", path="a.md", start_line=1, end_line=2, score=0.9),
        Passage(text="b", doc_id="b.md", path="b.md", start_line=5, end_line=8, score=0.5),
    ]
    ctx = _ctx(passages)
    cites = ctx.citations
    assert [c.index for c in cites] == [1, 2]
    assert cites[0].ref == "a.md:1-2"
    assert str(cites[0]) == "[1] a.md:1-2"
    assert cites[1].ref == "b.md:5-8"


def test_in_memory_passage_ref_uses_doc_id():
    p = Passage(text="hi", doc_id="mem:0", path=None, start_line=1, end_line=1, score=1.0)
    assert p.ref == "mem:0:1-1"


def test_to_text_includes_citation_lines():
    passages = [
        Passage(text="hello world", doc_id="a.md", path="a.md", start_line=1, end_line=1, score=1.0)
    ]
    ctx = _ctx(passages)
    out = ctx.to_text()
    assert "hello world" in out
    assert "[1] a.md:1-1" in out


def test_to_text_mark_wraps_highlights():
    p = Passage(
        text="hello world",
        doc_id="a.md",
        path="a.md",
        start_line=1,
        end_line=1,
        score=1.0,
        highlights=[(0, 5)],
    )
    out = _ctx([p]).to_text(mark=True)
    assert ">>hello<<" in out


def test_to_markdown_fences_code_paths():
    p = Passage(
        text="def f():\n    pass", doc_id="a.py", path="a.py", start_line=1, end_line=2, score=1.0
    )
    out = _ctx([p]).to_markdown()
    assert "```py" in out
    assert "def f():" in out


def test_to_markdown_quotes_prose_paths():
    p = Passage(text="hello world", doc_id="a.md", path="a.md", start_line=1, end_line=1, score=1.0)
    out = _ctx([p]).to_markdown()
    assert "> hello world" in out


def test_to_xml_wraps_passages_with_attributes():
    p = Passage(
        text="hello & world", doc_id="a.md", path="a.md", start_line=1, end_line=1, score=0.75
    )
    out = _ctx([p]).to_xml()
    assert out.startswith("<context>")
    assert out.endswith("</context>")
    assert 'index="1"' in out
    assert "hello &amp; world" in out  # XML-escaped


def test_to_xml_mark_inserts_mark_tags():
    p = Passage(
        text="hello world",
        doc_id="a.md",
        path="a.md",
        start_line=1,
        end_line=1,
        score=1.0,
        highlights=[(0, 5)],
    )
    out = _ctx([p]).to_xml(mark=True)
    assert "<mark>hello</mark>" in out


def test_to_json_round_trips_structure():
    p = Passage(
        text="hello world",
        doc_id="a.md",
        path="a.md",
        start_line=1,
        end_line=1,
        score=0.5,
        reasons=["direct match"],
        highlights=[(0, 5)],
    )
    payload = json.loads(_ctx([p]).to_json())
    assert payload["query"] == "installation steps"
    assert payload["passages"][0]["text"] == "hello world"
    assert payload["passages"][0]["highlights"] == [[0, 5]]
    assert payload["citations"] == ["[1] a.md:1-1"]


def test_empty_context_renders_without_error():
    ctx = _ctx([])
    assert ctx.to_text() == ""
    assert ctx.to_markdown() == ""
    assert ctx.to_xml() == "<context>\n\n</context>"
    assert json.loads(ctx.to_json())["passages"] == []
    assert ctx.citations == []

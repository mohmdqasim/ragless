from __future__ import annotations

from ragless.context.passage import Passage
from ragless.context.pipeline import (
    Compress,
    Dedup,
    Diversify,
    MergeAdjacent,
    Pack,
    PipelineState,
    WindowExpand,
    build_default_pipeline,
)


def _state(query="query", max_tokens=10_000, doc_text=None, token_counter=len, embed_fn=None):
    return PipelineState(
        query=query,
        max_tokens=max_tokens,
        token_counter=token_counter,
        doc_text=doc_text or (lambda doc_id: ""),
        embed_fn=embed_fn,
    )


def _passage(text="hello", doc_id="d", path="d.txt", start=1, end=1, score=1.0, **kw):
    return Passage(
        text=text, doc_id=doc_id, path=path, start_line=start, end_line=end, score=score, **kw
    )


# --- WindowExpand ------------------------------------------------------------


def test_window_expand_grows_line_range_from_doc_text():
    doc = "\n".join(f"line{i}" for i in range(1, 11))  # 10 lines
    state = _state(doc_text=lambda doc_id: doc)
    p = _passage(text="line5", start=5, end=5)
    out = WindowExpand(window=2).run([p], state)
    assert len(out) == 1
    assert out[0].start_line == 3
    assert out[0].end_line == 7
    assert out[0].text == "line3\nline4\nline5\nline6\nline7"
    assert "neighbor window" in out[0].reasons
    assert "direct match" in out[0].reasons


def test_window_expand_clamps_to_document_bounds():
    doc = "\n".join(f"line{i}" for i in range(1, 4))  # 3 lines
    state = _state(doc_text=lambda doc_id: doc)
    p = _passage(text="line1", start=1, end=1)
    out = WindowExpand(window=5).run([p], state)
    assert out[0].start_line == 1
    assert out[0].end_line == 3


def test_window_expand_zero_window_is_noop():
    state = _state(doc_text=lambda doc_id: "a\nb\nc")
    p = _passage(start=2, end=2)
    out = WindowExpand(window=0).run([p], state)
    assert out == [p]


def test_window_expand_missing_doc_text_leaves_passage_unchanged():
    state = _state(doc_text=lambda doc_id: "")
    p = _passage(start=1, end=1)
    out = WindowExpand(window=3).run([p], state)
    assert out[0] is p


# --- MergeAdjacent -------------------------------------------------------------


def test_merge_adjacent_overlapping_same_doc():
    doc = "\n".join(f"line{i}" for i in range(1, 11))
    state = _state(doc_text=lambda doc_id: doc)
    a = _passage(
        text="line1\nline2\nline3\nline4", start=1, end=4, score=0.5, reasons=["direct match"]
    )
    b = _passage(
        text="line3\nline4\nline5\nline6", start=3, end=6, score=0.9, reasons=["direct match"]
    )
    out = MergeAdjacent().run([a, b], state)
    assert len(out) == 1
    merged = out[0]
    assert merged.start_line == 1
    assert merged.end_line == 6
    assert merged.text == "line1\nline2\nline3\nline4\nline5\nline6"
    assert merged.score == 0.9
    assert merged.reasons == ["direct match"]


def test_merge_adjacent_respects_gap_parameter():
    doc = "\n".join(f"line{i}" for i in range(1, 21))
    state = _state(doc_text=lambda doc_id: doc)
    a = _passage(start=1, end=2)
    b = _passage(start=10, end=11)
    out = MergeAdjacent(gap=0).run([a, b], state)
    assert len(out) == 2  # far apart, not merged

    out2 = MergeAdjacent(gap=20).run([a, b], state)
    assert len(out2) == 1  # gap large enough to bridge them


def test_merge_adjacent_never_merges_across_documents():
    state = _state(doc_text=lambda doc_id: "\n".join(f"l{i}" for i in range(1, 11)))
    a = _passage(doc_id="doc_a", path="a.txt", start=1, end=2)
    b = _passage(doc_id="doc_b", path="b.txt", start=1, end=2)
    out = MergeAdjacent().run([a, b], state)
    assert len(out) == 2


def test_merge_adjacent_empty_input():
    state = _state()
    assert MergeAdjacent().run([], state) == []


# --- Dedup ---------------------------------------------------------------------


def test_dedup_drops_near_duplicate_keeping_higher_score():
    state = _state()
    a = _passage(text="the quick brown fox jumps over the lazy dog", score=0.4)
    b = _passage(text="the quick brown fox jumps over the lazy dog!", score=0.9)
    out = Dedup(threshold=90).run([a, b], state)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_dedup_keeps_distinct_passages():
    state = _state()
    a = _passage(text="completely different content about pandas")
    b = _passage(text="totally unrelated text about rockets")
    out = Dedup(threshold=92).run([a, b], state)
    assert len(out) == 2


def test_dedup_disabled_with_none_threshold():
    state = _state()
    a = _passage(text="same text")
    b = _passage(text="same text")
    out = Dedup(threshold=None).run([a, b], state)
    assert len(out) == 2


# --- Diversify (MMR) -------------------------------------------------------------


def test_diversify_noop_under_three_passages():
    state = _state()
    passages = [_passage(score=1.0), _passage(score=0.5)]
    assert Diversify().run(passages, state) == passages


def test_diversify_lexical_fallback_prefers_distinct_candidate_early():
    state = _state()
    a = _passage(text="pandas dataframe analysis tutorial", score=1.0)
    b = _passage(text="pandas dataframe analysis tutorial guide", score=0.95)  # near-dup of a
    c = _passage(text="completely unrelated rocket engine telemetry", score=0.5)
    out = Diversify(lambda_mult=0.3).run([a, b, c], state)
    assert len(out) == 3
    assert out[0].text == a.text  # most relevant goes first
    # the distinct passage should be preferred over the redundant near-duplicate for 2nd place
    assert out[1].text == c.text


def test_diversify_uses_embed_fn_when_provided():
    import numpy as np

    vocab = {"a": 0, "b": 1, "c": 2}

    def embed_fn(texts):
        vecs = np.zeros((len(texts), 3))
        for row, t in enumerate(texts):
            for word in t.split():
                if word in vocab:
                    vecs[row, vocab[word]] += 1.0
        return vecs

    state = _state(embed_fn=embed_fn)
    a = _passage(text="a a a", score=1.0)
    b = _passage(text="a a a", score=0.95)
    c = _passage(text="c c c", score=0.5)
    out = Diversify(lambda_mult=0.3).run([a, b, c], state)
    assert out[0].text == "a a a"
    assert out[1].text == "c c c"


# --- Compress --------------------------------------------------------------------


def test_compress_keeps_only_original_sentences():
    text = (
        "Ragless indexes a folder of files. "
        "It never calls a vector database. "
        "BM25 and fuzzy matching are combined with RRF. "
        "Nothing here is about weather."
    )
    p = _passage(text=text, path="readme.md")
    state = _state(query="vector database")
    out = Compress(ratio=0.3).run([p], state)
    compressed = out[0].text
    for line in compressed.split("\n"):
        line = line.strip()
        if line and line != "[...]":
            assert line in text  # never rewritten, only selected


def test_compress_code_file_compresses_by_lines():
    code = "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "",
            "def unrelated_noise():",
            "    pass",
            "",
            "def search(query):",
            "    return bm25.query(query)",
        ]
    )
    p = _passage(text=code, path="search.py")
    state = _state(query="search bm25 query")
    out = Compress(ratio=0.3).run([p], state)
    for line in out[0].text.split("\n"):
        line = line.strip()
        if line and line != "[...]":
            assert line in code


def test_compress_noop_when_query_has_no_content_terms():
    p = _passage(text="some sentence. another sentence.")
    state = _state(query="the a of")  # all stopwords
    out = Compress().run([p], state)
    assert out == [p]


# --- Pack ------------------------------------------------------------------------


def test_pack_never_exceeds_budget():
    state = _state(max_tokens=10, token_counter=len)
    passages = [_passage(text="12345"), _passage(text="12345"), _passage(text="12345")]
    out = Pack().run(passages, state)
    assert sum(len(p.text) for p in out) <= 10


def test_pack_budget_smaller_than_one_passage_returns_empty():
    state = _state(max_tokens=2, token_counter=len)
    passages = [_passage(text="way too long for the budget")]
    out = Pack().run(passages, state)
    assert out == []


def test_pack_empty_input():
    state = _state()
    assert Pack().run([], state) == []


# --- build_default_pipeline -------------------------------------------------------


def test_build_default_pipeline_respects_toggles():
    steps = build_default_pipeline(diversify=False, compress=False)
    kinds = [type(s).__name__ for s in steps]
    assert "Diversify" not in kinds
    assert "Compress" not in kinds
    assert kinds[-1] == "Pack"

    steps2 = build_default_pipeline(diversify=True, compress=True)
    kinds2 = [type(s).__name__ for s in steps2]
    assert "Diversify" in kinds2
    assert "Compress" in kinds2

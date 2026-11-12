from __future__ import annotations

from ragless.retrievers.bm25 import BM25Retriever
from ragless.types import Chunk


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(id=f"c{idx}", text=text, path=None, start_line=1, end_line=1)


def test_ranks_relevant_document_first():
    chunks = [
        _chunk(0, "the quick brown fox jumps over the lazy dog"),
        _chunk(1, "pandas dataframes support vectorized operations"),
        _chunk(2, "numpy arrays are the backbone of pandas"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.query("pandas dataframe", top_k=3)
    assert results
    assert results[0][0] in ("c1", "c2")
    result_ids = [r[0] for r in results]
    assert "c0" not in result_ids[:1]


def test_empty_index_returns_no_results():
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.query("anything", top_k=5) == []


def test_empty_query_returns_no_results():
    retriever = BM25Retriever()
    retriever.index([_chunk(0, "some text content")])
    assert retriever.query("", top_k=5) == []
    assert retriever.query("   ", top_k=5) == []


def test_no_matching_terms_returns_no_results():
    retriever = BM25Retriever()
    retriever.index([_chunk(0, "alpha beta gamma")])
    assert retriever.query("zzz totally unrelated query terms", top_k=5) == []


def test_top_k_is_respected():
    chunks = [_chunk(i, f"shared term plus unique{i}") for i in range(10)]
    retriever = BM25Retriever()
    retriever.index(chunks)
    results = retriever.query("shared term", top_k=3)
    assert len(results) == 3


def test_scores_are_sorted_descending():
    chunks = [
        _chunk(0, "python python python testing"),
        _chunk(1, "python testing"),
        _chunk(2, "testing"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    results = retriever.query("python testing", top_k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

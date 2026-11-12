from __future__ import annotations

from ragless.retrievers.fuzzy import FuzzyRetriever
from ragless.types import Chunk


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(id=f"c{idx}", text=text, path=None, start_line=1, end_line=1)


def test_handles_typo_queries():
    chunks = [
        _chunk(0, "installation instructions for ragless"),
        _chunk(1, "completely unrelated content about weather"),
    ]
    retriever = FuzzyRetriever()
    retriever.index(chunks)

    results = retriever.query("instalation instructons", top_k=2)  # typos
    assert results
    assert results[0][0] == "c0"


def test_empty_index_returns_no_results():
    retriever = FuzzyRetriever()
    retriever.index([])
    assert retriever.query("anything", top_k=5) == []


def test_empty_query_returns_no_results():
    retriever = FuzzyRetriever()
    retriever.index([_chunk(0, "some text")])
    assert retriever.query("", top_k=5) == []
    assert retriever.query("   ", top_k=5) == []


def test_top_k_is_respected():
    chunks = [_chunk(i, f"document number {i} about testing") for i in range(10)]
    retriever = FuzzyRetriever()
    retriever.index(chunks)
    results = retriever.query("document about testing", top_k=4)
    assert len(results) == 4

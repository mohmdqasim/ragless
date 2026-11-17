from __future__ import annotations

import time

import pytest

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


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        FuzzyRetriever(mode="nonsense")


def test_full_text_mode_still_handles_typos():
    chunks = [
        _chunk(0, "installation instructions for ragless"),
        _chunk(1, "completely unrelated content about weather"),
    ]
    retriever = FuzzyRetriever(mode="full_text")
    retriever.index(chunks)
    results = retriever.query("instalation instructons", top_k=2)
    assert results
    assert results[0][0] == "c0"


def test_vocabulary_mode_exact_term_match_scores_highest():
    chunks = [
        _chunk(0, "the retry policy uses exponential backoff"),
        _chunk(1, "retrying is common but this chunk is about something else entirely"),
    ]
    retriever = FuzzyRetriever()
    retriever.index(chunks)
    results = retriever.query("retry policy backoff", top_k=2)
    assert results
    assert results[0][0] == "c0"


def test_vocabulary_mode_does_not_match_unrelated_terms():
    chunks = [
        _chunk(0, "pandas dataframe analysis tutorial"),
        _chunk(1, "nothing to do with the query at all, just filler text here"),
    ]
    retriever = FuzzyRetriever()
    retriever.index(chunks)
    results = retriever.query("pandas dataframe", top_k=5)
    returned_ids = {chunk_id for chunk_id, _score in results}
    assert "c1" not in returned_ids


def test_vocabulary_mode_similarity_threshold_filters_distant_terms():
    chunks = [_chunk(0, "a chunk containing the word banana only")]
    retriever = FuzzyRetriever(similarity_threshold=95.0)
    retriever.index(chunks)
    # "zzzzz" is not close enough to "banana" at a 95 threshold
    assert retriever.query("zzzzz", top_k=5) == []


def test_vocabulary_mode_latency_stays_low_at_scale():
    chunks = [
        _chunk(i, f"document number {i} discusses retry policies and backoff schedules in detail")
        for i in range(1000)
    ]
    retriever = FuzzyRetriever()
    retriever.index(chunks)

    start = time.perf_counter()
    for _ in range(20):
        retriever.query("retry policy backoff shcedule", top_k=10)  # one typo
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / 20) * 1000
    assert avg_ms < 50  # generous: old full-text WRatio was ~70ms/query at this scale


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

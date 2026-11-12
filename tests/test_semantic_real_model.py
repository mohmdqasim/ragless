"""End-to-end tests with the real local embedding model and cross-encoder.

These download a small model on first run, so they're marked slow and are
skipped unless `pytest --run-slow` is passed. Requires `pip install ragless[semantic]`.
"""

from __future__ import annotations

import pytest

fastembed = pytest.importorskip("fastembed")

from ragless.core import Retriever  # noqa: E402


@pytest.mark.slow
def test_real_semantic_search_finds_conceptual_match():
    r = Retriever(
        [
            "Cats are small domesticated carnivorous mammals.",
            "The stock market fluctuated wildly this quarter.",
            "Kittens and felines make popular household pets.",
        ],
        engines=["semantic"],
    )
    results = r.search("domestic pets like cats", mode="semantic", top_k=2)
    assert results
    top_texts = [hit.text for hit in results[:2]]
    assert any("Cats" in t or "Kittens" in t for t in top_texts)


@pytest.mark.slow
def test_real_cross_encoder_rerank_runs():
    r = Retriever(["alpha document about pandas", "beta document about rockets"])
    results = r.search("pandas dataframe", rerank=True, top_k=2)
    assert results
    assert all("rerank" in hit.scores for hit in results)

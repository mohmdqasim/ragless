from __future__ import annotations

from ragless.core import Retriever


class _ReverseReranker:
    """A fake reranker that just reverses candidate order, for testing."""

    def rerank(self, query, chunks, top_k=None):
        reversed_chunks = list(reversed(chunks))
        n = len(reversed_chunks)
        pairs = [(c.id, float(n - i)) for i, c in enumerate(reversed_chunks)]
        return pairs[:top_k] if top_k is not None else pairs


def test_custom_reranker_controls_final_order():
    texts = ["alpha document", "beta document", "gamma document"]
    r = Retriever(texts)

    baseline = r.search("document", top_k=3)
    reranked = r.search("document", top_k=3, rerank=_ReverseReranker())

    assert [hit.chunk.id for hit in reranked] == list(reversed([hit.chunk.id for hit in baseline]))
    assert all("rerank" in hit.scores for hit in reranked)


def test_rerank_false_skips_reranking():
    texts = ["alpha document", "beta document"]
    r = Retriever(texts)
    results = r.search("document", rerank=False)
    assert all("rerank" not in hit.scores for hit in results)

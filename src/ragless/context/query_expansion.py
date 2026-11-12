"""Optional RM3-style pseudo-relevance feedback query expansion. Off by default."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ragless.context._lexical import content_terms
from ragless.types import SearchResult


def expand_query_terms(
    texts: Sequence[str], query: str, *, max_terms: int = 5, top_docs: int = 10
) -> list[str]:
    """Pick up to ``max_terms`` new terms from the top ``top_docs`` result texts.

    Returns an empty list if nothing clears the frequency bar or every
    frequent term is already in the query -- callers should fall back to the
    unexpanded query in that case.
    """
    query_terms = content_terms(query)
    freq: Counter[str] = Counter()
    for text in texts[:top_docs]:
        freq.update(content_terms(text))
    for term in query_terms:
        freq.pop(term, None)
    return [term for term, _count in freq.most_common(max_terms)]


def merge_search_results(
    a: Sequence[SearchResult], b: Sequence[SearchResult]
) -> list[SearchResult]:
    """Union two result lists by chunk id, keeping the higher-scored of each pair."""
    best: dict[str, SearchResult] = {}
    for r in (*a, *b):
        existing = best.get(r.chunk.id)
        if existing is None or r.score > existing.score:
            best[r.chunk.id] = r
    return sorted(best.values(), key=lambda r: r.score, reverse=True)

"""Fuzzy string-similarity retriever backed by rapidfuzz."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
from rapidfuzz import fuzz, process

from ragless.retrievers.base import BaseRetriever
from ragless.types import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FuzzyRetriever(BaseRetriever):
    """Ranks chunks by fuzzy string similarity to the query.

    Good at catching typos, partial matches, and near-duplicate phrasing that
    exact keyword search (BM25) would miss.

    Default (``mode="vocabulary"``): fuzzy matching happens against the
    corpus *vocabulary* (distinct terms), not every chunk's full text -- a
    query term either matches a vocabulary term exactly, or is mapped to the
    handful of vocabulary terms within ``similarity_threshold`` via
    ``rapidfuzz``, and matched terms score chunks BM25-style. This is much
    cheaper than comparing the whole query against every chunk, and more
    precise (a typo only pulls in chunks containing a genuinely close whole
    word, not chunks that merely share a long substring somewhere).

    ``mode="full_text"`` keeps the old behavior: ``scorer`` run directly
    against each chunk's full text via ``rapidfuzz.process.extract``.
    """

    name = "fuzzy"

    def __init__(
        self,
        scorer: Callable[..., float] = fuzz.ratio,
        *,
        mode: str = "vocabulary",
        similarity_threshold: float = 80.0,
        max_candidates_per_term: int = 5,
    ) -> None:
        if mode not in ("vocabulary", "full_text"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'vocabulary' or 'full_text'")
        self.scorer = scorer
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.max_candidates_per_term = max_candidates_per_term
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lens = np.zeros(0, dtype=np.float64)
        self._avgdl = 0.0
        self._idf: dict[str, float] = {}
        self._vocab: list[str] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunk_ids = [c.id for c in chunks]
        self._texts = [c.text for c in chunks]
        if self.mode == "full_text":
            return

        tokenized = [_tokenize(c.text) for c in chunks]
        self._doc_term_freqs = [Counter(toks) for toks in tokenized]
        self._doc_lens = np.array([len(toks) for toks in tokenized], dtype=np.float64)
        self._avgdl = float(self._doc_lens.mean()) if len(self._doc_lens) else 0.0

        n_docs = len(tokenized)
        doc_freq: Counter[str] = Counter()
        for toks in tokenized:
            doc_freq.update(set(toks))
        self._idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        self._vocab = sorted(doc_freq)

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self._chunk_ids or not query.strip():
            return []
        if self.mode == "full_text":
            return self._query_full_text(query, top_k)
        return self._query_vocabulary(query, top_k)

    def _query_full_text(self, query: str, top_k: int) -> list[tuple[str, float]]:
        matches = process.extract(query, self._texts, scorer=self.scorer, limit=top_k)
        return [(self._chunk_ids[index], float(score)) for _, score, index in matches if score > 0]

    def _query_vocabulary(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        matched_terms: dict[str, float] = {}
        for term in q_tokens:
            if term in self._idf:
                matched_terms[term] = max(matched_terms.get(term, 0.0), 100.0)
                continue
            if not self._vocab:
                continue
            for vocab_term, score, _idx in process.extract(
                term, self._vocab, scorer=self.scorer, limit=self.max_candidates_per_term
            ):
                if score >= self.similarity_threshold:
                    matched_terms[vocab_term] = max(
                        matched_terms.get(vocab_term, 0.0), float(score)
                    )

        if not matched_terms:
            return []

        n = len(self._chunk_ids)
        scores = np.zeros(n, dtype=np.float64)
        avgdl = self._avgdl or 1.0
        k1, b = 1.5, 0.75
        norm_lens = k1 * (1 - b + b * self._doc_lens / avgdl)

        for term, match_ratio in matched_terms.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self._doc_term_freqs], dtype=np.float64)
            denom = tf + norm_lens
            denom = np.where(denom == 0, 1.0, denom)
            weight = match_ratio / 100.0
            scores += weight * idf * (tf * (k1 + 1)) / denom

        hits = np.nonzero(scores > 0)[0]
        if hits.size == 0:
            return []
        k = min(top_k, hits.size)
        ranked = hits[np.argsort(-scores[hits])][:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in ranked]

"""BM25 (Okapi) keyword retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from ragless.retrievers.base import BaseRetriever
from ragless.types import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Classic Okapi BM25 ranking over tokenized chunk text.

    Pure NumPy implementation; no external search engine or index server.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunk_ids: list[str] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lens = np.zeros(0, dtype=np.float64)
        self._avgdl = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunk_ids = [c.id for c in chunks]
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

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self._chunk_ids:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        n = len(self._chunk_ids)
        scores = np.zeros(n, dtype=np.float64)
        avgdl = self._avgdl or 1.0
        norm_lens = self.k1 * (1 - self.b + self.b * self._doc_lens / avgdl)

        for term in q_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self._doc_term_freqs], dtype=np.float64)
            denom = tf + norm_lens
            denom = np.where(denom == 0, 1.0, denom)
            scores += idf * (tf * (self.k1 + 1)) / denom

        hits = np.nonzero(scores > 0)[0]
        if hits.size == 0:
            return []
        k = min(top_k, hits.size)
        ranked = hits[np.argsort(-scores[hits])][:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in ranked]

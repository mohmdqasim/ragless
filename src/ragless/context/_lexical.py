"""Small lexical helpers shared by compression, diversity fallback, and query expansion."""

from __future__ import annotations

import re
from collections.abc import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "but",
        "not",
        "can",
        "could",
        "do",
        "does",
        "did",
        "i",
        "you",
        "your",
        "we",
        "they",
        "their",
        "if",
        "then",
        "so",
        "such",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "why",
    }
)


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS}


def compute_highlights(text: str, terms: Iterable[str]) -> list[tuple[int, int]]:
    """Non-overlapping ``(start, end)`` offsets of ``terms`` found in ``text``."""
    ordered = sorted({t for t in terms if t}, key=len, reverse=True)
    if not ordered or not text:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in ordered) + r")\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]

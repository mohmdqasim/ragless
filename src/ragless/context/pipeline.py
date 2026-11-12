"""The composable context pipeline: expand -> merge -> dedup -> diversify -> compress -> pack."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from rapidfuzz import fuzz

from ragless.context._lexical import content_terms, tokenize
from ragless.context.passage import Passage

_CODE_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".kt",
    ".swift",
    ".cs",
}

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _is_code_path(path: str | None) -> bool:
    if not path:
        return False
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return False
    return f".{name.rsplit('.', 1)[-1].lower()}" in _CODE_EXTS


@dataclass
class PipelineState:
    """Shared, per-call context handed to every pipeline step."""

    query: str
    max_tokens: int
    token_counter: Callable[[str], int]
    doc_text: Callable[[str], str]
    """``doc_id -> full original text`` (``""`` if unavailable)."""
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None
    """Optional embedder used by :class:`Diversify`; falls back to lexical similarity."""


class ContextStep:
    """Base class for a pipeline step: takes passages, returns passages.

    Implementations should not mutate the input list or its ``Passage``
    objects in place (``Passage`` fields can be changed with
    ``dataclasses.replace``), so earlier steps' output stays inspectable.
    """

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        raise NotImplementedError


class WindowExpand(ContextStep):
    """Grows each passage to include ``window`` neighboring lines on each side."""

    def __init__(self, window: int = 2) -> None:
        self.window = window

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        if self.window <= 0:
            return passages
        out = []
        for p in passages:
            text = state.doc_text(p.doc_id)
            lines = text.splitlines()
            if not lines:
                out.append(p)
                continue
            new_start = max(1, p.start_line - self.window)
            new_end = min(len(lines), p.end_line + self.window)
            if new_start == p.start_line and new_end == p.end_line:
                out.append(p)
                continue
            new_text = "\n".join(lines[new_start - 1 : new_end])
            reasons = list(p.reasons)
            if "neighbor window" not in reasons:
                reasons.append("neighbor window")
            out.append(
                replace(p, text=new_text, start_line=new_start, end_line=new_end, reasons=reasons)
            )
        return out


def _merge_two(a: Passage, b: Passage, state: PipelineState) -> Passage:
    start, end = min(a.start_line, b.start_line), max(a.end_line, b.end_line)
    lines = state.doc_text(a.doc_id).splitlines()
    if lines and end <= len(lines):
        text = "\n".join(lines[start - 1 : end])
    else:
        text = a.text if len(a.text) >= len(b.text) else b.text
    reasons = list(dict.fromkeys([*a.reasons, *b.reasons]))
    return Passage(
        text=text,
        doc_id=a.doc_id,
        path=a.path,
        start_line=start,
        end_line=end,
        score=max(a.score, b.score),
        reasons=reasons,
        metadata={**b.metadata, **a.metadata},
    )


class MergeAdjacent(ContextStep):
    """Merges overlapping/adjacent passages from the same document into one."""

    def __init__(self, gap: int = 0) -> None:
        self.gap = gap

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        if not passages:
            return passages
        by_doc: dict[str, list[Passage]] = {}
        order: list[str] = []
        for p in passages:
            if p.doc_id not in by_doc:
                by_doc[p.doc_id] = []
                order.append(p.doc_id)
            by_doc[p.doc_id].append(p)

        merged: list[Passage] = []
        for doc_id in order:
            group = sorted(by_doc[doc_id], key=lambda p: p.start_line)
            current = group[0]
            for nxt in group[1:]:
                if nxt.start_line <= current.end_line + self.gap + 1:
                    current = _merge_two(current, nxt, state)
                else:
                    merged.append(current)
                    current = nxt
            merged.append(current)

        merged.sort(key=lambda p: p.score, reverse=True)
        return merged


class Dedup(ContextStep):
    """Drops near-duplicate passages (by fuzzy ratio), keeping the higher-scored one."""

    def __init__(self, threshold: float | None = 92.0) -> None:
        self.threshold = threshold

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        if not self.threshold or len(passages) < 2:
            return passages
        kept: list[Passage] = []
        for p in sorted(passages, key=lambda x: x.score, reverse=True):
            if not any(fuzz.ratio(p.text, k.text) >= self.threshold for k in kept):
                kept.append(p)
        return kept


def _similarity_matrix(passages: list[Passage], state: PipelineState) -> np.ndarray:
    n = len(passages)
    if state.embed_fn is not None:
        vecs = np.asarray(state.embed_fn([p.text for p in passages]), dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vecs / norms
        return np.clip(unit @ unit.T, 0.0, 1.0)

    sims = np.zeros((n, n))
    for i in range(n):
        sims[i, i] = 1.0
        for j in range(i + 1, n):
            s = fuzz.token_sort_ratio(passages[i].text, passages[j].text) / 100.0
            sims[i, j] = sims[j, i] = s
    return sims


class Diversify(ContextStep):
    """Reorders passages by Maximal Marginal Relevance to reduce redundancy."""

    def __init__(self, lambda_mult: float = 0.5) -> None:
        self.lambda_mult = lambda_mult

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        n = len(passages)
        if n < 3:
            return passages
        sims = _similarity_matrix(passages, state)
        scores = np.array([p.score for p in passages], dtype=np.float64)
        lo, hi = scores.min(), scores.max()
        relevance = (scores - lo) / (hi - lo) if hi > lo else np.ones(n)

        selected = [int(np.argmax(relevance))]
        remaining = [i for i in range(n) if i != selected[0]]
        while remaining:
            best_idx, best_val = remaining[0], -float("inf")
            for i in remaining:
                max_sim = max(sims[i, j] for j in selected)
                val = self.lambda_mult * relevance[i] - (1 - self.lambda_mult) * max_sim
                if val > best_val:
                    best_val, best_idx = val, i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [passages[i] for i in selected]


def _overlap_score(unit: str, query_terms: set[str]) -> int:
    return len(set(tokenize(unit)) & query_terms)


class Compress(ContextStep):
    """Extractive compression: keeps only the most query-relevant sentences/lines.

    Never rewrites text -- only selects a subsequence of the original
    sentences (prose) or lines (recognized code paths), joining
    non-contiguous kept spans with a literal ``[...]`` marker.
    """

    def __init__(self, ratio: float = 0.5, keep_neighbor: bool = True, min_keep: int = 1) -> None:
        self.ratio = ratio
        self.keep_neighbor = keep_neighbor
        self.min_keep = min_keep

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        query_terms = content_terms(state.query)
        if not query_terms:
            return passages
        return [self._compress_one(p, query_terms) for p in passages]

    def _compress_one(self, p: Passage, query_terms: set[str]) -> Passage:
        units = p.text.split("\n") if _is_code_path(p.path) else _SENT_RE.split(p.text)
        units = [u for u in units if u.strip()]
        if len(units) <= self.min_keep:
            return p

        scores = [_overlap_score(u, query_terms) for u in units]
        keep_n = max(self.min_keep, round(len(units) * self.ratio))
        keep_idx = set(sorted(range(len(units)), key=lambda i: scores[i], reverse=True)[:keep_n])

        if self.keep_neighbor:
            for i in list(keep_idx):
                if scores[i] > 0:
                    keep_idx.add(max(0, i - 1))
                    keep_idx.add(min(len(units) - 1, i + 1))

        kept_sorted = sorted(keep_idx)
        if not kept_sorted or len(kept_sorted) == len(units):
            return p

        pieces: list[str] = []
        prev: int | None = None
        for i in kept_sorted:
            if prev is not None and i != prev + 1:
                pieces.append("[...]")
            pieces.append(units[i])
            prev = i
        return replace(p, text="\n".join(pieces))


class Pack(ContextStep):
    """Greedily keeps whole passages (in their current order) under the token budget.

    Never exceeds ``state.max_tokens`` -- a passage that would overflow it is
    dropped rather than truncated mid-passage, so a citation is never cut off.
    """

    def run(self, passages: list[Passage], state: PipelineState) -> list[Passage]:
        kept: list[Passage] = []
        used = 0
        for p in passages:
            n = state.token_counter(p.text)
            if used + n <= state.max_tokens:
                kept.append(p)
                used += n
        return kept


def build_default_pipeline(
    *,
    window: int = 2,
    merge_gap: int = 0,
    dedup_threshold: float | None = 92.0,
    diversify: bool = True,
    mmr_lambda: float = 0.5,
    compress: bool = False,
    compress_ratio: float = 0.5,
) -> list[ContextStep]:
    steps: list[ContextStep] = [
        WindowExpand(window),
        MergeAdjacent(merge_gap),
        Dedup(dedup_threshold),
    ]
    if diversify:
        steps.append(Diversify(mmr_lambda))
    if compress:
        steps.append(Compress(compress_ratio))
    steps.append(Pack())
    return steps

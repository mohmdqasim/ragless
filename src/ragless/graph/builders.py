"""Pluggable edge builders: turn structure already in the files into typed, weighted edges."""

from __future__ import annotations

import ast
import os
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

from ragless.context._lexical import content_terms
from ragless.graph.edges import Edge
from ragless.types import Chunk

if TYPE_CHECKING:
    from ragless.embedders.base import Embedder


@dataclass
class BuildContext:
    """Everything an :class:`EdgeBuilder` needs to scan the corpus."""

    chunks: list[Chunk]
    chunk_doc_id: dict[str, str]
    doc_text: dict[str, str]
    chunks_by_doc: dict[str, list[Chunk]]
    only_source_paths: set[str] | None = None
    """When set, a builder only emits edges whose scan *originates* from one
    of these doc ids -- used for incremental updates. ``None`` means scan
    every document as a potential source."""

    def sources(self) -> list[str]:
        if self.only_source_paths is None:
            return list(self.chunks_by_doc)
        return [d for d in self.chunks_by_doc if d in self.only_source_paths]


@dataclass
class BuilderResult:
    edges: list[Edge] = field(default_factory=list)
    unresolved: dict[str, set[str]] = field(default_factory=dict)
    """source doc id -> targets it tried to resolve but couldn't."""


def make_build_context(
    chunks: Sequence[Chunk],
    chunk_doc_id: dict[str, str],
    doc_text: dict[str, str],
    *,
    only_source_paths: set[str] | None = None,
) -> BuildContext:
    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[chunk_doc_id[c.id]].append(c)
    for doc_chunks in chunks_by_doc.values():
        doc_chunks.sort(key=lambda c: c.start_line)
    return BuildContext(
        chunks=list(chunks),
        chunk_doc_id=chunk_doc_id,
        doc_text=doc_text,
        chunks_by_doc=dict(chunks_by_doc),
        only_source_paths=only_source_paths,
    )


def _chunk_for_line(doc_chunks: list[Chunk], line: int) -> Chunk | None:
    for c in doc_chunks:
        if c.start_line <= line <= c.end_line:
            return c
    return None


class EdgeBuilder(ABC):
    """A pluggable strategy for discovering edges between chunks.

    ``build()`` must never raise on unresolvable references -- it should
    just skip them (recording them in ``BuilderResult.unresolved`` when the
    reference looked like it was meant to resolve within the corpus), so one
    broken link or unparseable file never breaks indexing.
    """

    name: str
    weight: float = 1.0

    @abstractmethod
    def build(self, ctx: BuildContext) -> BuilderResult:
        raise NotImplementedError


# --- Links -------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s][^)]*)\)")
_MD_REF_USE_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")
_MD_REF_DEF_RE = re.compile(r"^\s*\[([^\]]+)\]:\s*(\S+)", re.MULTILINE)
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|[^\]]*)?\]\]")
_BARE_PATH_RE = re.compile(r"(?<![\w/])(?:\.{1,2}/[\w./-]+|[\w-]+(?:/[\w.-]+)+\.[A-Za-z0-9]{1,5})")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text).strip("-")


class LinkEdges(EdgeBuilder):
    """Markdown links, reference-style links, wiki-links, and bare relative paths."""

    name = "link"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def build(self, ctx: BuildContext) -> BuilderResult:
        known = set(p for p in ctx.doc_text if not p.startswith("mem:"))
        result = BuilderResult()

        for doc_id in ctx.sources():
            if doc_id.startswith("mem:"):
                continue
            text = ctx.doc_text.get(doc_id, "")
            if not text:
                continue
            doc_chunks = ctx.chunks_by_doc.get(doc_id, [])
            targets: list[tuple[int, str, str | None]] = []  # (line, target, anchor)
            consumed_spans: list[tuple[int, int]] = []

            ref_defs = dict(_MD_REF_DEF_RE.findall(text))
            for pattern, has_anchor_group in (
                (_MD_LINK_RE, False),
                (_WIKI_LINK_RE, True),
            ):
                for m in pattern.finditer(text):
                    consumed_spans.append(m.span())
                    line = text.count("\n", 0, m.start()) + 1
                    if has_anchor_group:
                        targets.append((line, m.group(1), m.group(2)))
                    else:
                        targets.append((line, m.group(2), None))
            for m in _MD_REF_USE_RE.finditer(text):
                ref = (m.group(2) or m.group(1)).strip().lower()
                target = ref_defs.get(ref) or ref_defs.get(m.group(1).strip().lower())
                if target:
                    consumed_spans.append(m.span())
                    line = text.count("\n", 0, m.start()) + 1
                    targets.append((line, target, None))

            unresolved_here: set[str] = set()
            for line, raw_target, anchor in targets:
                target = raw_target.strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "#", "//")):
                    continue
                target, _, inline_anchor = target.partition("#")
                anchor = anchor or inline_anchor or None
                if not target:
                    continue
                resolved = _resolve_relative(doc_id, target, known)
                if resolved is None or resolved == doc_id:
                    if resolved is None:
                        unresolved_here.add(target)
                    continue
                src_chunk = _chunk_for_line(doc_chunks, line) or (
                    doc_chunks[0] if doc_chunks else None
                )
                dst_chunks = ctx.chunks_by_doc.get(resolved, [])
                dst_chunk = dst_chunks[0] if dst_chunks else None
                if anchor and dst_chunks:
                    dst_chunk = (
                        _find_heading_chunk(ctx.doc_text[resolved], dst_chunks, anchor) or dst_chunk
                    )
                if src_chunk is None or dst_chunk is None:
                    continue
                result.edges.append(
                    Edge(
                        src=src_chunk.id,
                        dst=dst_chunk.id,
                        type=self.name,
                        weight=self.weight,
                        reason=f"{doc_id} links to {resolved}",
                    )
                )

            # bare relative paths mentioned in prose, outside of [...] link syntax
            for m in _BARE_PATH_RE.finditer(text):
                if any(m.start() < end and start < m.end() for start, end in consumed_spans):
                    continue
                raw = m.group(0)
                resolved = _resolve_relative(doc_id, raw, known)
                if resolved is None or resolved == doc_id:
                    continue
                line = text.count("\n", 0, m.start()) + 1
                src_chunk = _chunk_for_line(doc_chunks, line) or (
                    doc_chunks[0] if doc_chunks else None
                )
                dst_chunks = ctx.chunks_by_doc.get(resolved, [])
                if src_chunk is None or not dst_chunks:
                    continue
                result.edges.append(
                    Edge(
                        src=src_chunk.id,
                        dst=dst_chunks[0].id,
                        type=self.name,
                        weight=self.weight * 0.8,
                        reason=f"{doc_id} mentions {resolved}",
                    )
                )

            if unresolved_here:
                result.unresolved[doc_id] = unresolved_here
        return result


def _resolve_relative(source_path: str, target: str, known: set[str]) -> str | None:
    source_dir = str(PurePosixPath(source_path).parent)
    if "/" in target or target.startswith((".", "..")):
        candidate = os.path.normpath(os.path.join(source_dir, target))
        if candidate in known:
            return candidate
        stem_matches = [p for p in known if PurePosixPath(p).name == PurePosixPath(candidate).name]
    else:
        same_dir = os.path.normpath(os.path.join(source_dir, target))
        if same_dir in known:
            return same_dir
        stem_matches = [
            p
            for p in known
            if PurePosixPath(p).name == target
            or PurePosixPath(p).stem.lower() == PurePosixPath(target).stem.lower()
        ]
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def _find_heading_chunk(doc_text: str, doc_chunks: list[Chunk], anchor: str) -> Chunk | None:
    anchor_slug = _slugify(anchor)
    for m in _HEADING_RE.finditer(doc_text):
        if _slugify(m.group(2)) == anchor_slug:
            line = doc_text.count("\n", 0, m.start()) + 1
            return _chunk_for_line(doc_chunks, line)
    return None


# --- Imports -------------------------------------------------------------------

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[\w*${},\s]+from\s+)?|export\s+[\w*${},\s]+from\s+|require\()\s*['"]([^'"]+)['"]"""
)


class ImportEdges(EdgeBuilder):
    """Python imports (via ``ast``) and JS/TS imports/requires (via regex)."""

    name = "import"

    def __init__(self, weight: float = 0.8) -> None:
        self.weight = weight

    def build(self, ctx: BuildContext) -> BuilderResult:
        result = BuilderResult()
        py_paths = {p for p in ctx.doc_text if p.endswith(".py")}
        js_exts = (".js", ".jsx", ".ts", ".tsx")
        js_paths = {p for p in ctx.doc_text if p.endswith(js_exts)}

        for doc_id in ctx.sources():
            if doc_id.endswith(".py"):
                self._build_python(doc_id, ctx, py_paths, result)
            elif doc_id.endswith(js_exts):
                self._build_js(doc_id, ctx, js_paths, result)
        return result

    def _build_python(
        self, doc_id: str, ctx: BuildContext, py_paths: set[str], result: BuilderResult
    ) -> None:
        text = ctx.doc_text.get(doc_id, "")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return
        doc_chunks = ctx.chunks_by_doc.get(doc_id, [])
        unresolved_here: set[str] = set()

        for node in ast.walk(tree):
            # (resolved_path_or_None, display_name_for_unresolved)
            attempts: list[tuple[str | None, str]] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    attempts.append(
                        (_resolve_python_import(doc_id, alias.name, 0, py_paths), alias.name)
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module:
                    # `from . import a, b` -- no module to resolve as itself (that
                    # would just be the package dir); each name may be its own
                    # submodule file, e.g. `a` -> `pkg/a.py`.
                    for alias in node.names:
                        sub_display = f"{'.' * node.level}{alias.name}"
                        sub_resolved = _resolve_python_import(
                            doc_id, alias.name, node.level, py_paths
                        )
                        attempts.append((sub_resolved, sub_display))
                else:
                    display = f"{'.' * node.level}{module}"
                    attempts.append(
                        (_resolve_python_import(doc_id, module, node.level, py_paths), display)
                    )
            else:
                continue

            for resolved, display in attempts:
                if resolved is None:
                    unresolved_here.add(display)
                    continue
                if resolved == doc_id:
                    continue
                src_chunk = _chunk_for_line(doc_chunks, node.lineno) or (
                    doc_chunks[0] if doc_chunks else None
                )
                dst_chunks = ctx.chunks_by_doc.get(resolved, [])
                if src_chunk is None or not dst_chunks:
                    continue
                result.edges.append(
                    Edge(
                        src=src_chunk.id,
                        dst=dst_chunks[0].id,
                        type=self.name,
                        weight=self.weight,
                        reason=f"{doc_id} imports {resolved}",
                    )
                )
        if unresolved_here:
            result.unresolved[doc_id] = unresolved_here

    def _build_js(
        self, doc_id: str, ctx: BuildContext, js_paths: set[str], result: BuilderResult
    ) -> None:
        text = ctx.doc_text.get(doc_id, "")
        doc_chunks = ctx.chunks_by_doc.get(doc_id, [])
        unresolved_here: set[str] = set()

        for m in _JS_IMPORT_RE.finditer(text):
            spec = m.group(1)
            if not spec.startswith("."):
                continue  # bare package specifiers are never resolvable within the corpus
            resolved = _resolve_js_import(doc_id, spec, js_paths)
            if resolved is None or resolved == doc_id:
                if resolved is None:
                    unresolved_here.add(spec)
                continue
            line = text.count("\n", 0, m.start()) + 1
            src_chunk = _chunk_for_line(doc_chunks, line) or (doc_chunks[0] if doc_chunks else None)
            dst_chunks = ctx.chunks_by_doc.get(resolved, [])
            if src_chunk is None or not dst_chunks:
                continue
            result.edges.append(
                Edge(
                    src=src_chunk.id,
                    dst=dst_chunks[0].id,
                    type=self.name,
                    weight=self.weight,
                    reason=f"{doc_id} imports {resolved}",
                )
            )
        if unresolved_here:
            result.unresolved[doc_id] = unresolved_here


def _resolve_python_import(
    source_path: str, module: str, level: int, py_paths: set[str]
) -> str | None:
    if level > 0:
        base = PurePosixPath(source_path).parent
        for _ in range(level - 1):
            base = base.parent
        parts = module.split(".") if module else []
        candidate = base.joinpath(*parts) if parts else base
        for suffix in (f"{candidate}.py", str(PurePosixPath(candidate) / "__init__.py")):
            norm = os.path.normpath(suffix)
            if norm in py_paths:
                return norm
        return None

    parts = module.split(".") if module else []
    if not parts:
        return None
    candidates = ["/".join(parts) + ".py", "/".join(parts) + "/__init__.py"]
    for cand in candidates:
        matches = [p for p in py_paths if p == cand or p.endswith("/" + cand)]
        if len(matches) == 1:
            return matches[0]
    return None


def _resolve_js_import(source_path: str, spec: str, js_paths: set[str]) -> str | None:
    source_dir = str(PurePosixPath(source_path).parent)
    base = os.path.normpath(os.path.join(source_dir, spec))
    candidates = [base] + [f"{base}{ext}" for ext in (".ts", ".tsx", ".js", ".jsx")]
    candidates += [
        str(PurePosixPath(base) / f"index{ext}") for ext in (".ts", ".tsx", ".js", ".jsx")
    ]
    for cand in candidates:
        if cand in js_paths:
            return cand
    return None


# --- Symbols -------------------------------------------------------------------

_JS_DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\s+([A-Za-z_]\w*)"
    r"|class\s+([A-Za-z_]\w*)"
    r"|const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\()",
    re.MULTILINE,
)

_COMMON_SYMBOL_NAMES = frozenset(
    {"self", "cls", "main", "init", "run", "get", "set", "data", "value"}
)


class SymbolEdges(EdgeBuilder):
    """Connects a symbol's definition chunk to chunks that reference it elsewhere.

    This builder is corpus-wide by nature (a usage can be anywhere), so it is
    treated as a "global" builder for cache invalidation purposes: it is
    always fully recomputed, never incrementally patched.
    """

    name = "symbol"

    def __init__(
        self,
        weight: float = 0.6,
        min_name_length: int = 3,
        max_usage_fraction: float = 0.3,
        max_usage_edges: int = 25,
    ) -> None:
        self.weight = weight
        self.min_name_length = min_name_length
        self.max_usage_fraction = max_usage_fraction
        self.max_usage_edges = max_usage_edges

    def build(self, ctx: BuildContext) -> BuilderResult:
        result = BuilderResult()
        definitions: list[tuple[str, Chunk]] = []  # (name, defining chunk)

        for doc_id, doc_chunks in ctx.chunks_by_doc.items():
            if doc_id.endswith(".py"):
                definitions.extend(
                    self._python_defs(doc_id, ctx.doc_text.get(doc_id, ""), doc_chunks)
                )
            else:
                definitions.extend(self._regex_defs(doc_chunks))

        if not definitions:
            return result

        n_chunks = len(ctx.chunks)
        max_usage_chunks = max(1, int(n_chunks * self.max_usage_fraction))

        for name, def_chunk in definitions:
            if len(name) < self.min_name_length or name in _COMMON_SYMBOL_NAMES:
                continue
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            counts: list[tuple[int, Chunk]] = []
            for c in ctx.chunks:
                if c.id == def_chunk.id:
                    continue
                n = len(pattern.findall(c.text))
                if n:
                    counts.append((n, c))
            if not counts or len(counts) > max_usage_chunks:
                continue
            counts.sort(key=lambda pair: pair[0], reverse=True)
            for _count, usage_chunk in counts[: self.max_usage_edges]:
                result.edges.append(
                    Edge(
                        src=def_chunk.id,
                        dst=usage_chunk.id,
                        type=self.name,
                        weight=self.weight,
                        reason=f"defines {name!r}, used here",
                    )
                )
        return result

    def _python_defs(
        self, doc_id: str, text: str, doc_chunks: list[Chunk]
    ) -> list[tuple[str, Chunk]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        out = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk = _chunk_for_line(doc_chunks, node.lineno)
                if chunk is not None:
                    out.append((node.name, chunk))
        return out

    def _regex_defs(self, doc_chunks: list[Chunk]) -> list[tuple[str, Chunk]]:
        out = []
        for chunk in doc_chunks:
            for m in _JS_DEF_RE.finditer(chunk.text):
                name = next(g for g in m.groups() if g)
                out.append((name, chunk))
        return out


# --- Structure -----------------------------------------------------------------


class StructureEdges(EdgeBuilder):
    """Sequential neighbor chunks and markdown heading hierarchy, within one file."""

    name = "structure"

    def __init__(self, weight: float = 0.3) -> None:
        self.weight = weight

    def build(self, ctx: BuildContext) -> BuilderResult:
        result = BuilderResult()
        for doc_id in ctx.sources():
            doc_chunks = ctx.chunks_by_doc.get(doc_id, [])
            for a, b in zip(doc_chunks, doc_chunks[1:], strict=False):
                result.edges.append(
                    Edge(
                        src=a.id,
                        dst=b.id,
                        type=self.name,
                        weight=self.weight,
                        reason=f"sequential neighbor in {doc_id}",
                    )
                )
            result.edges.extend(
                self._heading_edges(doc_id, ctx.doc_text.get(doc_id, ""), doc_chunks)
            )
        return result

    def _heading_edges(self, doc_id: str, text: str, doc_chunks: list[Chunk]) -> list[Edge]:
        stack: list[tuple[int, Chunk, str]] = []
        edges: list[Edge] = []
        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))
            title = m.group(2)
            line = text.count("\n", 0, m.start()) + 1
            chunk = _chunk_for_line(doc_chunks, line)
            if chunk is None:
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack and stack[-1][1].id != chunk.id:
                parent_level, parent_chunk, parent_title = stack[-1]
                edges.append(
                    Edge(
                        src=parent_chunk.id,
                        dst=chunk.id,
                        type=self.name,
                        weight=self.weight,
                        reason=f"heading hierarchy: {parent_title!r} > {title!r}",
                    )
                )
            stack.append((level, chunk, title))
        return edges


# --- Shared rare terms -----------------------------------------------------------


class SharedTermEdges(EdgeBuilder):
    """Connects chunks sharing distinctive (high-IDF) terms. Corpus-wide; always rebuilt."""

    name = "shared_term"

    def __init__(
        self,
        weight: float = 0.4,
        terms_per_chunk: int = 6,
        min_overlap: int = 2,
        max_edges_per_node: int = 8,
    ) -> None:
        self.weight = weight
        self.terms_per_chunk = terms_per_chunk
        self.min_overlap = min_overlap
        self.max_edges_per_node = max_edges_per_node

    def build(self, ctx: BuildContext) -> BuilderResult:
        result = BuilderResult()
        chunks = ctx.chunks
        n = len(chunks)
        if n < 2:
            return result

        doc_freq: Counter[str] = Counter()
        chunk_terms: list[set[str]] = []
        for c in chunks:
            terms = content_terms(c.text)
            chunk_terms.append(terms)
            doc_freq.update(terms)

        idf = {t: np.log(n / df) for t, df in doc_freq.items()}
        distinctive: list[set[str]] = []
        for terms in chunk_terms:
            ranked = sorted(terms, key=lambda t: idf.get(t, 0.0), reverse=True)
            distinctive.append(set(ranked[: self.terms_per_chunk]))

        inverted: dict[str, list[int]] = defaultdict(list)
        for i, terms in enumerate(distinctive):
            for t in terms:
                inverted[t].append(i)

        overlap_counts: Counter[tuple[int, int]] = Counter()
        shared_terms: dict[tuple[int, int], set[str]] = defaultdict(set)
        for term, idxs in inverted.items():
            if len(idxs) < 2 or len(idxs) > max(10, n // 4):
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pair = (idxs[a], idxs[b])
                    overlap_counts[pair] += 1
                    shared_terms[pair].add(term)

        candidate_edges: list[tuple[float, int, int]] = []
        for (i, j), count in overlap_counts.items():
            if count < self.min_overlap:
                continue
            avg_idf = sum(idf.get(t, 0.0) for t in shared_terms[(i, j)]) / count
            score = count * avg_idf
            candidate_edges.append((score, i, j))
        candidate_edges.sort(reverse=True)

        edge_count: Counter[int] = Counter()
        for score, i, j in candidate_edges:
            if edge_count[i] >= self.max_edges_per_node or edge_count[j] >= self.max_edges_per_node:
                continue
            terms_preview = ", ".join(sorted(shared_terms[(i, j)])[:3])
            weight = min(
                1.0, score / (self.terms_per_chunk * max(idf.values(), default=1.0) or 1.0)
            )
            result.edges.append(
                Edge(
                    src=chunks[i].id,
                    dst=chunks[j].id,
                    type=self.name,
                    weight=self.weight * max(weight, 0.1),
                    reason=f"shares distinctive terms: {terms_preview}",
                )
            )
            edge_count[i] += 1
            edge_count[j] += 1
        return result


# --- Semantic kNN -----------------------------------------------------------------


class SemanticKNNEdges(EdgeBuilder):
    """Connects each chunk to its top-k most similar chunks by embedding cosine similarity.

    Only useful (and only added to the default builder set) when the
    ``semantic`` extra is installed and an embedder is available.
    """

    name = "semantic_knn"

    def __init__(
        self,
        embedder: Embedder,
        weight: float = 0.5,
        k: int = 5,
        min_similarity: float = 0.5,
        batch_size: int = 256,
    ) -> None:
        self.embedder = embedder
        self.weight = weight
        self.k = k
        self.min_similarity = min_similarity
        self.batch_size = batch_size

    def build(self, ctx: BuildContext) -> BuilderResult:
        result = BuilderResult()
        chunks = ctx.chunks
        n = len(chunks)
        if n < 2:
            return result

        vectors = self.embedder.embed([c.text for c in chunks]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms

        seen: set[tuple[str, str]] = set()
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            sims = unit[start:end] @ unit.T
            for row_offset, row in enumerate(sims):
                i = start + row_offset
                row = row.copy()
                row[i] = -1.0
                top_idx = np.argpartition(-row, min(self.k, n - 1) - 1)[: self.k]
                top_idx = top_idx[np.argsort(-row[top_idx])]
                for j in top_idx:
                    sim = float(row[j])
                    if sim < self.min_similarity:
                        continue
                    key = (
                        (chunks[i].id, chunks[int(j)].id)
                        if chunks[i].id < chunks[int(j)].id
                        else (
                            chunks[int(j)].id,
                            chunks[i].id,
                        )
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    result.edges.append(
                        Edge(
                            src=chunks[i].id,
                            dst=chunks[int(j)].id,
                            type=self.name,
                            weight=self.weight * sim,
                            reason=f"semantically similar (cosine={sim:.2f})",
                        )
                    )
        return result


def build_default_edge_builders(embedder: Embedder | None = None) -> list[EdgeBuilder]:
    """The default edge-builder set: everything except semantic kNN, which needs an embedder."""
    builders: list[EdgeBuilder] = [
        LinkEdges(),
        ImportEdges(),
        SymbolEdges(),
        StructureEdges(),
        SharedTermEdges(),
    ]
    if embedder is not None:
        builders.append(SemanticKNNEdges(embedder))
    return builders


LOCAL_BUILDER_NAMES = frozenset({"link", "import", "structure"})
GLOBAL_BUILDER_NAMES = frozenset({"symbol", "shared_term", "semantic_knn"})


def builder_fingerprint(builders: Sequence[EdgeBuilder]) -> str:
    parts = [f"{b.name}:{type(b).__name__}:{getattr(b, 'weight', '')}" for b in builders]
    return "|".join(parts)

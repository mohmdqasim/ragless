"""The context pipeline's data model: passages, citations, and the final result."""

from __future__ import annotations

import json
import xml.sax.saxutils as xml_escape
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)


@dataclass
class Passage:
    """One piece of context on its way through the pipeline.

    Unlike :class:`~ragless.types.SearchResult`, a passage may have been
    expanded, merged with a neighbor, or compressed -- ``text`` and the line
    range always describe what the passage looks like *right now*, not the
    original chunk it started from.
    """

    text: str
    doc_id: str
    """Internal document key: the file path, or ``mem:<n>`` for in-memory text."""
    path: str | None
    start_line: int
    end_line: int
    score: float
    reasons: list[str] = field(default_factory=lambda: ["direct match"])
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: list[tuple[int, int]] = field(default_factory=list)
    """Character offsets of matched query terms within ``text``, ``(start, end)``."""
    chunk_id: str | None = None
    """The single original chunk id this passage came from, if it still maps to
    exactly one -- cleared (``None``) once a passage has been merged/expanded
    across chunk boundaries, since it no longer corresponds to just one."""

    @property
    def ref(self) -> str:
        """A citation reference like ``docs/auth.md:42-58`` or ``mem:0:1-3``."""
        if self.path is not None:
            return f"{self.path}:{self.start_line}-{self.end_line}"
        return f"{self.doc_id}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class Citation:
    """A numbered reference to one passage, e.g. ``[1] docs/auth.md:42-58``."""

    index: int
    ref: str
    path: str | None
    start_line: int
    end_line: int
    score: float
    reasons: list[str]

    def __str__(self) -> str:
        return f"[{self.index}] {self.ref}"


def _marked(text: str, highlights: list[tuple[int, int]], before: str, after: str) -> str:
    if not highlights:
        return text
    spans = sorted(highlights)
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(before)
        out.append(text[start:end])
        out.append(after)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


@dataclass
class Context:
    """The finished, budget-fitted context for a query: ready to hand to any LLM."""

    query: str
    passages: list[Passage]
    token_count: int

    @property
    def citations(self) -> list[Citation]:
        return [
            Citation(
                index=i,
                ref=p.ref,
                path=p.path,
                start_line=p.start_line,
                end_line=p.end_line,
                score=p.score,
                reasons=list(p.reasons),
            )
            for i, p in enumerate(self.passages, start=1)
        ]

    def to_text(self, *, mark: bool = False) -> str:
        """Plain text: one passage per block, separated by blank lines, with
        a trailing ``[n] ref`` citation line for each."""
        blocks = []
        for i, p in enumerate(self.passages, start=1):
            body = _marked(p.text, p.highlights, ">>", "<<") if mark else p.text
            blocks.append(f"{body}\n[{i}] {p.ref}")
        return "\n\n".join(blocks)

    def to_markdown(self, *, mark: bool = False) -> str:
        """Markdown: a fenced block per passage (fenced as code for code paths,
        quoted otherwise) with a numbered reference heading."""
        from ragless.context.pipeline import _is_code_path

        blocks = []
        for i, p in enumerate(self.passages, start=1):
            body = _marked(p.text, p.highlights, "**", "**") if mark else p.text
            heading = f"**[{i}] {p.ref}**"
            if _is_code_path(p.path):
                lang = (p.path or "").rsplit(".", 1)[-1]
                block = f"{heading}\n```{lang}\n{body}\n```"
            else:
                quoted = "\n".join(f"> {line}" for line in body.splitlines()) or ">"
                block = f"{heading}\n{quoted}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def to_xml(self, *, mark: bool = False) -> str:
        """XML-tagged blocks: ``<passage index="1" ref="...">text</passage>``."""
        blocks = []
        for i, p in enumerate(self.passages, start=1):
            body = _marked(p.text, p.highlights, "<mark>", "</mark>") if mark else p.text
            escaped = xml_escape.escape(body) if not mark else _xml_escape_outside_marks(body)
            blocks.append(
                f'<passage index="{i}" ref="{xml_escape.quoteattr(p.ref)[1:-1]}" '
                f'score="{p.score:.6g}">{escaped}</passage>'
            )
        return "<context>\n" + "\n".join(blocks) + "\n</context>"

    def to_json(self) -> str:
        """JSON: full structured output, including highlight offsets."""
        payload = {
            "query": self.query,
            "token_count": self.token_count,
            "passages": [
                {
                    "text": p.text,
                    "doc_id": p.doc_id,
                    "path": p.path,
                    "start_line": p.start_line,
                    "end_line": p.end_line,
                    "score": p.score,
                    "reasons": p.reasons,
                    "metadata": p.metadata,
                    "highlights": [list(h) for h in p.highlights],
                }
                for p in self.passages
            ],
            "citations": [str(c) for c in self.citations],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _xml_escape_outside_marks(text: str) -> str:
    # `_marked` already inserted literal <mark>/</mark>; escape everything else.
    parts = text.split("<mark>")
    out = [xml_escape.escape(parts[0])]
    for part in parts[1:]:
        marked, _, rest = part.partition("</mark>")
        out.append(f"<mark>{xml_escape.escape(marked)}</mark>")
        out.append(xml_escape.escape(rest))
    return "".join(out)

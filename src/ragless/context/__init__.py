"""ragless.context: turn search results into ready-to-use, budget-fitted context.

No LLM calls happen anywhere in this package -- it only builds the context
you hand to whichever model you call yourself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from ragless.context._lexical import compute_highlights, content_terms
from ragless.context.passage import DEFAULT_PROMPT_TEMPLATE, Citation, Context, Passage
from ragless.context.pipeline import (
    Compress,
    ContextStep,
    Dedup,
    Diversify,
    MergeAdjacent,
    Pack,
    PipelineState,
    WindowExpand,
    build_default_pipeline,
)
from ragless.context.query_expansion import expand_query_terms, merge_search_results
from ragless.context.tokens import TiktokenCounter, estimate_tokens
from ragless.types import SearchResult

__all__ = [
    "Context",
    "Citation",
    "Passage",
    "ContextStep",
    "WindowExpand",
    "MergeAdjacent",
    "Dedup",
    "Diversify",
    "Compress",
    "Pack",
    "PipelineState",
    "build_default_pipeline",
    "build_context",
    "render_prompt",
    "estimate_tokens",
    "TiktokenCounter",
    "expand_query_terms",
    "merge_search_results",
    "DEFAULT_PROMPT_TEMPLATE",
]


def build_context(
    *,
    query: str,
    results: Sequence[SearchResult],
    doc_text: Callable[[str], str],
    chunk_doc_id: Callable[[str], str],
    max_tokens: int,
    token_counter: Callable[[str], int] | None = None,
    pipeline: Sequence[ContextStep] | None = None,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    highlight: bool = True,
    window: int = 2,
    merge_gap: int = 0,
    dedup_threshold: float | None = 92.0,
    diversify: bool = True,
    mmr_lambda: float = 0.5,
    compress: bool = False,
    compress_ratio: float = 0.5,
) -> Context:
    """Run ``results`` through the context pipeline and pack them into a :class:`Context`.

    This is what :meth:`ragless.Retriever.context` calls; it is exposed
    directly for callers who already have a custom result set (e.g. merged
    from a retriever they built themselves) and want to skip straight to
    packing it.
    """
    passages = [
        Passage(
            text=r.chunk.text,
            doc_id=chunk_doc_id(r.chunk.id),
            path=r.chunk.path,
            start_line=r.chunk.start_line,
            end_line=r.chunk.end_line,
            score=r.score,
            metadata=dict(r.chunk.metadata),
        )
        for r in results
    ]

    state = PipelineState(
        query=query,
        max_tokens=max_tokens,
        token_counter=token_counter or estimate_tokens,
        doc_text=doc_text,
        embed_fn=embed_fn,
    )

    steps = (
        list(pipeline)
        if pipeline is not None
        else build_default_pipeline(
            window=window,
            merge_gap=merge_gap,
            dedup_threshold=dedup_threshold,
            diversify=diversify,
            mmr_lambda=mmr_lambda,
            compress=compress,
            compress_ratio=compress_ratio,
        )
    )
    for step in steps:
        passages = step.run(passages, state)

    if highlight:
        terms = content_terms(query)
        passages = [replace(p, highlights=compute_highlights(p.text, terms)) for p in passages]

    token_count = sum(state.token_counter(p.text) for p in passages)
    return Context(query=query, passages=passages, token_count=token_count)


def render_prompt(
    query: str,
    ctx: Context,
    *,
    template: str | Callable[[str, Context], str] | None = None,
    context_format: str = "markdown",
) -> str:
    """Render ``ctx`` into a prompt string.

    ``template`` is either a ``str.format``-style template with
    ``{query}``/``{context}`` placeholders, or a callable ``(query, ctx) ->
    str`` for full control. ``context_format`` picks which ``Context.to_*()``
    method fills in ``{context}`` for the string-template case.
    """
    if callable(template):
        return template(query, ctx)

    renderers = {
        "text": ctx.to_text,
        "markdown": ctx.to_markdown,
        "xml": ctx.to_xml,
        "json": ctx.to_json,
    }
    if context_format not in renderers:
        raise ValueError(
            f"Unknown context_format {context_format!r}; expected one of {sorted(renderers)}"
        )
    tmpl = template or DEFAULT_PROMPT_TEMPLATE
    return tmpl.format(query=query, context=renderers[context_format]())

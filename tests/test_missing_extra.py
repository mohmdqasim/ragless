"""Verifies a clear error when the optional 'semantic' extra isn't installed.

Simulates the missing dependency by making `import fastembed` fail, rather
than requiring an environment without fastembed actually installed.
"""

from __future__ import annotations

import sys

import pytest

from ragless.exceptions import MissingDependencyError


def _block_import(monkeypatch, module_name: str):
    monkeypatch.setitem(sys.modules, module_name, None)


def test_fastembed_embedder_raises_clear_error(monkeypatch):
    _block_import(monkeypatch, "fastembed")
    sys.modules.pop("ragless.embedders.fastembed_embedder", None)

    from ragless.embedders.fastembed_embedder import FastEmbedEmbedder

    with pytest.raises(MissingDependencyError) as exc_info:
        FastEmbedEmbedder()
    assert "ragless[semantic]" in str(exc_info.value)


def test_cross_encoder_reranker_raises_clear_error(monkeypatch):
    _block_import(monkeypatch, "fastembed.rerank.cross_encoder")
    sys.modules.pop("ragless.rerank", None)

    from ragless.rerank import CrossEncoderReranker

    with pytest.raises(MissingDependencyError) as exc_info:
        CrossEncoderReranker()
    assert "ragless[semantic]" in str(exc_info.value)


def test_retriever_semantic_engine_without_extra_raises(monkeypatch):
    _block_import(monkeypatch, "fastembed")
    sys.modules.pop("ragless.embedders.fastembed_embedder", None)
    sys.modules.pop("ragless.core", None)

    from ragless.core import Retriever

    with pytest.raises(MissingDependencyError):
        Retriever(["some text"], engines=["semantic"])

from __future__ import annotations

import sys

import pytest

from ragless.context.tokens import TiktokenCounter, estimate_tokens
from ragless.exceptions import MissingDependencyError


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_tokens_roughly_four_chars_per_token():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("hi") == 1  # at least one token for any non-empty text


def test_estimate_tokens_grows_with_length():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_tiktoken_counter_missing_extra_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingDependencyError) as exc_info:
        TiktokenCounter()
    assert "ragless[tokens]" in str(exc_info.value)


@pytest.mark.slow
def test_tiktoken_counter_real_encoding_counts_tokens():
    counter = TiktokenCounter()
    assert counter("") == 0
    assert counter("hello world") > 0
    assert counter("hello world, this is a longer sentence") > counter("hello world")

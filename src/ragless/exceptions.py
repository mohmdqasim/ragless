"""Exceptions raised by ragless."""

from __future__ import annotations


class RaglessError(Exception):
    """Base class for all ragless errors."""


class MissingDependencyError(RaglessError):
    """Raised when an optional extra is required but not installed."""

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{extra}' extra. "
            f"Install it with: pip install 'ragless[{extra}]'"
        )


class UnknownRetrieverError(RaglessError):
    """Raised when an unrecognised retriever name is requested."""

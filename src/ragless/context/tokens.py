"""Token counting for context packing: a free default, and an exact extra."""

from __future__ import annotations

from ragless.exceptions import MissingDependencyError


def estimate_tokens(text: str) -> int:
    """Dependency-free token estimate (~4 characters per token).

    Good enough for budget packing without pulling in a tokenizer; use
    :class:`TiktokenCounter` or your own callable when exact counts matter.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class TiktokenCounter:
    """Exact token counts via ``tiktoken``. Requires ``pip install ragless[tokens]``."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised without extra installed
            raise MissingDependencyError("Exact token counting", "tokens") from exc

        self._encoding = tiktoken.get_encoding(encoding)

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

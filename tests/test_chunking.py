from __future__ import annotations

import pytest

from ragless.chunking import LineChunker


def test_single_small_chunk():
    chunker = LineChunker(lines_per_chunk=40, overlap=5)
    text = "line one\nline two\nline three"
    chunks = chunker.chunk(text, path="a.txt", chunk_id_prefix="a.txt")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == text
    assert chunk.path == "a.txt"
    assert chunk.start_line == 1
    assert chunk.end_line == 3
    assert chunk.id == "a.txt:0"


def test_splits_into_overlapping_windows():
    chunker = LineChunker(lines_per_chunk=10, overlap=2)
    lines = [f"line {i}" for i in range(25)]
    text = "\n".join(lines)
    chunks = chunker.chunk(text, path="big.txt", chunk_id_prefix="big.txt")

    # windows: [1-10], [9-18], [17-25]
    assert [c.start_line for c in chunks] == [1, 9, 17]
    assert [c.end_line for c in chunks] == [10, 18, 25]
    # overlap: chunk 2 should repeat the tail of chunk 1
    assert chunks[0].text.splitlines()[-2:] == chunks[1].text.splitlines()[:2]


def test_empty_text_produces_no_chunks():
    chunker = LineChunker()
    assert chunker.chunk("", path="empty.txt", chunk_id_prefix="empty.txt") == []
    assert chunker.chunk("   \n  \n", path="blank.txt", chunk_id_prefix="blank.txt") == []


def test_metadata_is_attached_to_every_chunk():
    chunker = LineChunker(lines_per_chunk=5, overlap=1)
    text = "\n".join(f"l{i}" for i in range(12))
    chunks = chunker.chunk(
        text, path="m.txt", chunk_id_prefix="m.txt", metadata={"category": "guide"}
    )
    assert len(chunks) > 1
    assert all(c.metadata == {"category": "guide"} for c in chunks)


def test_in_memory_chunk_has_no_path():
    chunker = LineChunker()
    chunks = chunker.chunk("hello world", path=None, chunk_id_prefix="mem:0")
    assert chunks[0].path is None
    assert chunks[0].id == "mem:0:0"


@pytest.mark.parametrize(
    "lines_per_chunk,overlap",
    [(0, 0), (-1, 0), (10, 10), (10, 11), (5, -1)],
)
def test_invalid_parameters_raise(lines_per_chunk, overlap):
    with pytest.raises(ValueError):
        LineChunker(lines_per_chunk=lines_per_chunk, overlap=overlap)

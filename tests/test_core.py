from __future__ import annotations

import pytest

from ragless.core import Retriever
from ragless.exceptions import UnknownRetrieverError
from ragless.retrievers import BM25Retriever, FuzzyRetriever, SemanticRetriever


def test_search_finds_relevant_markdown_file(sample_corpus):
    r = Retriever(str(sample_corpus))
    results = r.search("cached embeddings npy json sidecar", mode="hybrid")
    assert results
    assert results[0].path.endswith("caching.md")
    assert results[0].start_line >= 1
    assert results[0].end_line >= results[0].start_line


def test_default_engines_are_bm25_and_fuzzy_only(sample_corpus):
    r = Retriever(str(sample_corpus))
    assert set(r._retrievers.keys()) == {"bm25", "fuzzy"}


def test_single_engine_modes(sample_corpus):
    r = Retriever(str(sample_corpus))
    bm25_only = r.search("installation", mode="bm25")
    fuzzy_only = r.search("instalation", mode="fuzzy")  # typo
    assert bm25_only
    assert fuzzy_only


def test_unknown_mode_raises(sample_corpus):
    r = Retriever(str(sample_corpus))
    with pytest.raises(UnknownRetrieverError):
        r.search("query", mode="nonexistent")


def test_in_memory_text_search():
    r = Retriever(
        [
            "The capital of France is Paris.",
            "The capital of Japan is Tokyo.",
        ]
    )
    results = r.search("capital of Japan")
    assert results
    assert results[0].path is None
    assert "Tokyo" in results[0].text


def test_combined_source_and_texts(sample_corpus):
    r = Retriever(str(sample_corpus), texts=["extra in-memory note about retries"])
    results = r.search("retries", mode="hybrid")
    assert results
    assert any(hit.path is None for hit in results)


def test_empty_directory_returns_no_results(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    r = Retriever(str(empty_dir))
    assert len(r) == 0
    assert r.search("anything") == []


def test_nonexistent_source_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        Retriever(str(missing))


def test_binary_files_are_skipped(tmp_path):
    (tmp_path / "readable.txt").write_text("some readable content about dogs")
    (tmp_path / "image.bin").write_bytes(bytes([0, 1, 2, 0, 255, 254, 0, 0, 10, 20]))
    r = Retriever(str(tmp_path))
    paths = {chunk.path for chunk in r._chunks}
    assert any(p.endswith("readable.txt") for p in paths)
    assert not any(p.endswith("image.bin") for p in paths)


def test_empty_file_produces_no_chunks(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "content.txt").write_text("some content here")
    r = Retriever(str(tmp_path))
    assert len(r) > 0
    paths = {chunk.path for chunk in r._chunks}
    assert not any(p.endswith("empty.txt") for p in paths)


def test_unicode_file_is_indexed_and_searchable(tmp_path):
    (tmp_path / "unicode.txt").write_text(
        "café naïve 你好 \U0001f600 résumé unique-marker-42",
        encoding="utf-8",
    )
    r = Retriever(str(tmp_path))
    results = r.search("unique-marker-42")
    assert results
    assert "unique-marker-42" in results[0].text


def test_very_large_file_indexes_without_error(tmp_path):
    lines = [f"line {i} contains some repeated filler words for bulk testing" for i in range(20000)]
    lines[15000] = "line 15000 contains the special needle phrase zzzyyyxxx"
    (tmp_path / "huge.txt").write_text("\n".join(lines))

    r = Retriever(str(tmp_path))
    assert len(r) > 1
    results = r.search("special needle phrase zzzyyyxxx", mode="bm25")
    assert results
    assert "zzzyyyxxx" in results[0].text


def test_include_exclude_filters(tmp_path):
    (tmp_path / "a.md").write_text("markdown content about testing")
    (tmp_path / "b.py").write_text("python content about testing")
    r = Retriever(str(tmp_path), include=["*.md"])
    paths = {chunk.path for chunk in r._chunks}
    assert any(p.endswith("a.md") for p in paths)
    assert not any(p.endswith("b.py") for p in paths)

    r2 = Retriever(str(tmp_path), exclude=["*.py"])
    paths2 = {chunk.path for chunk in r2._chunks}
    assert any(p.endswith("a.md") for p in paths2)
    assert not any(p.endswith("b.py") for p in paths2)


def test_path_filter_on_search(sample_corpus):
    r = Retriever(str(sample_corpus))
    results = r.search("cached embedded", mode="bm25", path_filter="*guides*", top_k=50)
    assert results
    assert all(hit.path and "guides" in hit.path for hit in results)


def test_path_filter_excludes_in_memory_chunks():
    r = Retriever(["memory text about testing"])
    assert r.search("testing", path_filter="*.md") == []


def test_metadata_filters(tmp_path):
    (tmp_path / "guide.txt").write_text("guide content about onboarding")
    (tmp_path / "reference.txt").write_text("reference content about onboarding")

    def tag(path):
        if path and "guide" in path:
            return {"category": "guide"}
        return {"category": "reference"}

    r = Retriever(str(tmp_path), metadata=tag)
    results = r.search("onboarding", filters={"category": "guide"}, top_k=50)
    assert results
    assert all(hit.metadata.get("category") == "guide" for hit in results)


def test_metadata_filter_with_predicate(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content shared keyword")
    (tmp_path / "b.txt").write_text("beta content shared keyword")
    r = Retriever(str(tmp_path), metadata=lambda path: {"length": len(path or "")})
    results = r.search("shared keyword", filters={"length": lambda v: v > 0}, top_k=50)
    assert results


def test_top_k_limits_results(sample_corpus):
    r = Retriever(str(sample_corpus))
    results = r.search("content", mode="bm25", top_k=1)
    assert len(results) <= 1


def test_custom_engines_mapping_with_semantic(fake_embedder):
    texts = ["pandas dataframe tutorial", "completely unrelated weather report"]
    r = Retriever(
        texts,
        engines={
            "bm25": BM25Retriever(),
            "fuzzy": FuzzyRetriever(),
            "semantic": SemanticRetriever(fake_embedder),
        },
    )
    assert set(r._retrievers.keys()) == {"bm25", "fuzzy", "semantic"}
    results = r.search("pandas dataframe", mode="semantic")
    assert results[0].scores.get("semantic") is not None


def test_weights_and_rrf_k_are_configurable(sample_corpus):
    r = Retriever(str(sample_corpus), weights={"bm25": 2.0, "fuzzy": 0.5}, rrf_k=10)
    results = r.search("installation", mode="hybrid")
    assert results

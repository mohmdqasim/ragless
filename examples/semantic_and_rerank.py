"""Semantic search with on-disk embedding caching and cross-encoder reranking.

Requires `pip install ragless[semantic]`. The first run downloads a small
local model; later runs reuse the cached embeddings for unchanged files.

Run from the repo root: `python examples/semantic_and_rerank.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"
CACHE_DIR = Path(__file__).parent / ".ragless_cache"


def main() -> None:
    retriever = Retriever(
        str(DOCS_DIR),
        engines=["bm25", "fuzzy", "semantic"],
        cache_dir=str(CACHE_DIR),
    )

    query = "how are embeddings kept up to date"
    print(f"Query: {query!r}\n")

    print("-- hybrid (bm25 + fuzzy + semantic) --")
    for hit in retriever.search(query, mode="hybrid", top_k=3):
        print(f"  {hit.path}:{hit.start_line}-{hit.end_line}  score={hit.score:.4f}")

    print("\n-- hybrid + cross-encoder rerank --")
    for hit in retriever.search(query, mode="hybrid", top_k=3, rerank=True):
        print(f"  {hit.path}:{hit.start_line}-{hit.end_line}  score={hit.score:.4f}")


if __name__ == "__main__":
    main()

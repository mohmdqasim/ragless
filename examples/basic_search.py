"""Index a folder of markdown docs and search it with hybrid (BM25 + fuzzy) search.

Run from the repo root: `python examples/basic_search.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"


def main() -> None:
    retriever = Retriever(str(DOCS_DIR))
    print(f"Indexed {len(retriever)} chunks from {DOCS_DIR}\n")

    for query in ["how do I install the semantic extra", "line numbers in results"]:
        print(f"Query: {query!r}")
        for hit in retriever.search(query, mode="hybrid", top_k=3):
            print(f"  {hit.path}:{hit.start_line}-{hit.end_line}  score={hit.score:.4f}")
            print(f"    {hit.text.strip().splitlines()[0]}")
        print()


if __name__ == "__main__":
    main()

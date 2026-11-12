"""Filter search results by metadata or by file path glob.

Run from the repo root: `python examples/filters_and_metadata.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"


def tag_by_path(path: str | None) -> dict:
    if path and "guides" in path:
        return {"category": "guide"}
    return {"category": "reference"}


def main() -> None:
    retriever = Retriever(str(DOCS_DIR), metadata=tag_by_path)

    print("-- only chunks tagged 'guide' --")
    for hit in retriever.search("embeddings", filters={"category": "guide"}):
        print(f"  {hit.path}  category={hit.metadata['category']}")

    print("\n-- only chunks from files under guides/ --")
    for hit in retriever.search("embeddings", path_filter="*guides*"):
        print(f"  {hit.path}")


if __name__ == "__main__":
    main()

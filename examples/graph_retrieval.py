"""Graph-based retrieval: follow links/imports/symbols to pull in related chunks.

No LLM is involved -- the graph is built from structure already in the files.

Run from the repo root: `python examples/graph_retrieval.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"


def main() -> None:
    retriever = Retriever(str(DOCS_DIR))

    print("-- graph stats --")
    print(retriever.graph.stats())

    query = "how do I search once ragless is installed"
    print(f"\nQuery: {query!r}\n")

    print("-- hybrid search alone --")
    for hit in retriever.search(query, top_k=3):
        print(f"  {hit.path}  score={hit.score:.4f}")

    print("\n-- hybrid search + graph=True --")
    for hit in retriever.search(query, top_k=3, graph=True):
        print(f"  {hit.path}  score={hit.score:.4f}  reasons={hit.graph_reasons}")

    print("\n-- context(graph=True) --")
    ctx = retriever.context(query, max_tokens=500, graph=True)
    for passage in ctx.passages:
        print(f"  {passage.path}  reasons={passage.reasons}")


if __name__ == "__main__":
    main()

"""Build ready-to-use context for any LLM -- no vector database, no LLM call here.

Run from the repo root: `python examples/context_for_llm.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"


def main() -> None:
    retriever = Retriever(str(DOCS_DIR))
    query = "how are embeddings cached"

    ctx = retriever.context(query, max_tokens=300)
    print(f"Packed {len(ctx.passages)} passage(s), ~{ctx.token_count} tokens\n")

    print("-- markdown --")
    print(ctx.to_markdown())

    print("\n-- citations --")
    for citation in ctx.citations:
        print(f"  {citation}  score={citation.score:.4f}  reasons={citation.reasons}")

    print("\n-- plain text with matched terms marked --")
    print(ctx.to_text(mark=True))


if __name__ == "__main__":
    main()

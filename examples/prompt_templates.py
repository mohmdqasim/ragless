"""Render a ready-to-send prompt string -- ragless never calls an LLM itself.

Run from the repo root: `python examples/prompt_templates.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"


def main() -> None:
    retriever = Retriever(str(DOCS_DIR))
    query = "how do I install the semantic extra"

    print("-- default template --")
    print(retriever.prompt(query, max_tokens=300))

    print("\n-- custom string template --")
    custom = "Context:\n{context}\n\nUser question: {query}\nRespond concisely:"
    print(retriever.prompt(query, template=custom, max_tokens=300))

    print("\n-- custom callable template --")
    print(
        retriever.prompt(
            query,
            template=lambda q, ctx: f"[{len(ctx.passages)} passage(s)] {q}",
            max_tokens=300,
        )
    )


if __name__ == "__main__":
    main()

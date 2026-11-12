"""Search a list of in-memory strings -- no files required.

Run from the repo root: `python examples/in_memory_text.py`
"""

from ragless import Retriever

NOTES = [
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "Mount Everest is the tallest mountain on Earth.",
]


def main() -> None:
    retriever = Retriever(NOTES)
    for hit in retriever.search("capital of Japan"):
        print(f"score={hit.score:.4f}  {hit.text}")


if __name__ == "__main__":
    main()

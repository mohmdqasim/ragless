"""Export the retrieval graph and visualize it -- JSON/GraphML need no extra
dependency; a quick degree-based text visualization works without one either.
A real image render is offered if `networkx` + `matplotlib` happen to be
installed (`pip install ragless[graph] matplotlib`), but is entirely optional.

Run from the repo root: `python examples/graph_export.py`
"""

from pathlib import Path

from ragless import Retriever

DOCS_DIR = Path(__file__).parent / "sample_docs"
OUT_DIR = Path(__file__).parent / ".graph_export"


def text_visualization(graph) -> str:
    """A dependency-free visualization: each file, its related files, ranked by weight."""
    paths = sorted(set(graph.chunk_doc_id.values()))
    lines = []
    for path in paths:
        related = graph.related(path, k=3)
        if not related:
            continue
        lines.append(Path(path).name + ":")
        for other, weight in related:
            lines.append(f"  -- {weight:.2f} --> {Path(other).name}")
    return "\n".join(lines) if lines else "(no edges)"


def main() -> None:
    retriever = Retriever(str(DOCS_DIR))
    graph = retriever.graph

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "graph.json").write_text(graph.to_json())
    (OUT_DIR / "graph.graphml").write_text(graph.to_graphml())
    print(f"Wrote {OUT_DIR / 'graph.json'} and {OUT_DIR / 'graph.graphml'}")

    print("\n-- text visualization (always available) --")
    print(text_visualization(graph))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        nx_graph = graph.to_networkx()
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(nx_graph, seed=0)
        nx.draw(nx_graph, pos, ax=ax, node_size=200, with_labels=False, arrows=True)
        fig.savefig(OUT_DIR / "graph.png", dpi=150)
        print(f"\nAlso wrote {OUT_DIR / 'graph.png'} (networkx + matplotlib were available)")
    except ImportError:
        print("\n(Skipping graph.png: install `ragless[graph]` and `matplotlib` to render one.)")


if __name__ == "__main__":
    main()

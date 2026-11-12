"""Honest recall/precision and timing numbers for graph expansion.

Marked slow: not required for the main coverage gate, but run with
`pytest --run-slow -m slow tests/test_graph_benchmark.py -s` to print the
real numbers that get copied into the README by hand.
"""

from __future__ import annotations

import random
import time

import pytest

from ragless import Retriever

pytestmark = pytest.mark.slow


def _multi_hop_corpus(tmp_path):
    """Docs/queries where the keyword match lands on a *pointer* file, but the
    actually useful content is only in a *linked* file -- the case graph
    expansion is built for. Returns (corpus_dir, list[(query, expected_path_suffix)]).
    """
    docs = tmp_path / "corpus"
    docs.mkdir()

    (docs / "auth.md").write_text(
        "# Auth\n\nUsers log in with a password. See [retry policy](./retry.md) "
        "for how failed logins are retried.\n"
    )
    (docs / "retry.md").write_text(
        "# Retry policy\n\nExponential backoff schedule: wait 1s, 2s, 4s, 8s, "
        "then give up and surface an error to the caller.\n"
    )
    (docs / "support.md").write_text(
        "# Support runbook\n\nFor billing questions about refunds see policy.md directly.\n"
    )
    (docs / "policy.md").write_text(
        "# Refund policy\n\nRefunds are issued within fourteen days of the original purchase.\n"
    )
    pkg = docs / "pkg"
    pkg.mkdir()
    (pkg / "client.py").write_text(
        "from pkg.util import backoff\n\n\ndef call():\n    return backoff(3)\n"
    )
    (pkg / "util.py").write_text(
        "def backoff(attempt):\n    return 2 ** attempt\n\n\n"
        "class RetryPolicy:\n    def wait(self, attempt):\n        return backoff(attempt)\n"
    )
    # distractors: no links, no shared vocabulary with the queries below
    (docs / "weather.md").write_text(
        "Weather note: cloudy with a chance of rain in the mountains.\n"
    )
    (docs / "cooking.md").write_text("Add salt and simmer the sauce for ten minutes on low heat.\n")
    # a case graph expansion can't help: isolated, directly matched, no edges at all
    (docs / "isolated.md").write_text(
        "# Standalone note\n\nThis file links to nothing and is linked by nothing.\n"
    )

    queries = [
        ("how does login retry work", "retry.md"),
        ("refund timing for a purchase", "policy.md"),
        ("what does the exponential backoff wait schedule look like", "util.py"),
        ("standalone note content", "isolated.md"),  # the honest "graph doesn't help" case
    ]
    return docs, queries


def _recall_precision_at_k(retriever, queries, *, k, graph):
    hits = 0
    for query, expected_suffix in queries:
        results = retriever.search(query, top_k=k, fetch_k=1, graph=graph)
        found = any(h.path and h.path.endswith(expected_suffix) for h in results)
        hits += int(found)
    n = len(queries)
    recall = hits / n
    precision = hits / (n * k)
    return recall, precision


def test_recall_precision_with_and_without_graph(tmp_path, capsys):
    docs, queries = _multi_hop_corpus(tmp_path)
    r = Retriever(str(docs))

    k = 3
    recall_plain, precision_plain = _recall_precision_at_k(r, queries, k=k, graph=False)
    recall_graph, precision_graph = _recall_precision_at_k(r, queries, k=k, graph=True)

    with capsys.disabled():
        print(f"\nrecall@{k}   without graph: {recall_plain:.2f}   with graph: {recall_graph:.2f}")
        print(f"precision@{k} without: {precision_plain:.3f}   with graph: {precision_graph:.3f}")

    # graph expansion must not make the multi-hop queries *worse*
    assert recall_graph >= recall_plain
    # the isolated-note query is the honest "graph can't help" case: unaffected
    isolated_plain = any(
        h.path and h.path.endswith("isolated.md")
        for h in r.search("standalone note content", top_k=k, fetch_k=1, graph=False)
    )
    isolated_graph = any(
        h.path and h.path.endswith("isolated.md")
        for h in r.search("standalone note content", top_k=k, fetch_k=1, graph=True)
    )
    assert isolated_plain == isolated_graph


def _synthetic_corpus(tmp_path, n_files=200, lines_per_file=25, link_probability=0.3):
    """~5,000 chunks (n_files * lines_per_file / ~1 chunk per short file, so
    scale n_files*lines_per_file to roughly the target chunk count via a small
    chunker). Returns the corpus dir."""
    docs = tmp_path / "synthetic"
    docs.mkdir()
    rng = random.Random(42)
    names = [f"doc_{i:04d}.md" for i in range(n_files)]
    for i, name in enumerate(names):
        lines = [f"# Document {i}", ""]
        for j in range(lines_per_file):
            lines.append(f"Line {j} of document {i} talks about topic number {(i * 7 + j) % 50}.")
        if rng.random() < link_probability:
            target = rng.choice(names)
            if target != name:
                lines.append(f"\nSee also [related document](./{target}) for more context.")
        (docs / name).write_text("\n".join(lines) + "\n")
    return docs


def test_graph_build_and_query_latency_on_synthetic_corpus(tmp_path, capsys):
    from ragless.chunking import LineChunker

    docs = _synthetic_corpus(tmp_path, n_files=700, lines_per_file=25)

    start = time.perf_counter()
    r = Retriever(str(docs), chunker=LineChunker(lines_per_chunk=5, overlap=1))
    chunk_count = len(r)
    index_time = time.perf_counter() - start

    start = time.perf_counter()
    graph = r.graph
    build_time = time.perf_counter() - start

    queries = [f"topic number {i}" for i in (3, 17, 29, 41)]
    latencies = []
    for q in queries:
        start = time.perf_counter()
        r.search(q, top_k=10, graph=True, hops=2)
        latencies.append(time.perf_counter() - start)
    avg_query_latency = sum(latencies) / len(latencies)

    with capsys.disabled():
        print(f"\nsynthetic corpus: {chunk_count} chunks, {len(graph.edges)} edges")
        print(f"indexing time: {index_time:.3f}s")
        print(f"graph build time: {build_time:.3f}s")
        print(f"avg graph-expanded query latency: {avg_query_latency * 1000:.1f}ms")

    assert chunk_count > 1000  # in the right ballpark for the "~5k chunks" target
    assert build_time < 30.0  # generous upper bound so this isn't flaky in CI
    assert avg_query_latency < 5.0

"""Runs each example script's main() to make sure the README/examples stay accurate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _load_example(name: str):
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "basic_search.py",
        "filters_and_metadata.py",
        "in_memory_text.py",
        "context_for_llm.py",
        "prompt_templates.py",
    ],
)
def test_fast_example_runs_without_error(name, capsys):
    module = _load_example(name)
    module.main()
    assert capsys.readouterr().out.strip()


@pytest.mark.slow
def test_semantic_example_runs_without_error(capsys):
    module = _load_example("semantic_and_rerank.py")
    module.main()
    assert capsys.readouterr().out.strip()

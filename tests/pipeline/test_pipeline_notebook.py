"""The version 2 pipeline notebook, and the 1.x Extract-Modeling notebook cut to its model cells.

Every step cell of ``MoSeq2-Pipeline.ipynb`` is two lines, the step's import from
``moseq2_app.notebook`` and its call. The notebooks carry one version. No cell of
the cut 1.x notebooks calls a notebook function stage 2 removed.
"""

from __future__ import annotations

import json
import re

import pytest

from cells import NOTEBOOKS

STEPS = ("setup", "arena_tool", "extract", "diagnostics", "train_pca", "scores_and_changepoints")

#: The 1.x notebook functions stage 2's first pull request removes.
REMOVED = ("extract_found_sessions", "extract_command", "aggregate_extract_results_command", "get_selected_sessions",
           "batch_extract", "run_local_extract", "aggregate_results")


def _code(name):
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_each_step_cell_is_two_lines():
    cells = _code("MoSeq2-Pipeline.ipynb")

    assert cells == [f"from moseq2_app.notebook import {step}\n{step}()" for step in STEPS]


def test_the_notebook_version_matches_the_others():
    versions = {path.name: json.loads(path.read_text(encoding="utf-8"))["metadata"].get("moseq2", {}).get("version")
                for path in NOTEBOOKS.glob("*.ipynb")}

    assert versions["MoSeq2-Pipeline.ipynb"] is not None
    assert set(versions.values()) == {versions["MoSeq2-Pipeline.ipynb"]}


@pytest.mark.parametrize("name", ["MoSeq2-Extract-Modeling-Notebook.ipynb",
                                  "[OPTIONAL]MoSeq2-Apply-Pre-Trained-Model-Notebook.ipynb"])
def test_the_cut_notebook_calls_no_removed_function(name):
    code = "\n".join(_code(name))

    for function in REMOVED:
        assert not re.search(rf"\b{function}\b", code), function

"""The notebook's Extract step extracts the number of frames config.yaml sets.

``num_frames`` in ``config.yaml`` limits each extraction to its first N frames.
``moseq2 extract`` in the project honours it; the notebook's Extract step runs
the same step on the same file and must extract the same frames.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py

from extract_projects import small_project

SESSION = "saline_example_0"
NUM_FRAMES = 100


def test_notebook_batch_extraction_honours_num_frames(tmp_path, raw_sessions):
    from moseq2_app.notebook import extract

    project = small_project(tmp_path / "project", raw_sessions, (SESSION,), num_frames=NUM_FRAMES)

    before = os.getcwd()
    os.chdir(project.folder)
    try:
        widget = extract()
        widget.start()
        run = widget.wait()
    finally:
        os.chdir(before)
    assert run["state"] == "finished", widget.problem

    with h5py.File(Path(project.folder) / "aggregate_results" / SESSION / "results.h5", "r") as f:
        assert f["frames"].shape[0] == NUM_FRAMES
        assert f["timestamps"].shape[0] == NUM_FRAMES

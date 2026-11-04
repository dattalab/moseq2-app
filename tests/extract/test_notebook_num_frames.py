"""The notebook's batch extraction extracts the number of frames config.yaml sets.

``num_frames`` in ``config.yaml`` limits each extraction to its first N frames.
``moseq2-extract batch-extract --config-file config.yaml`` honours it; the
notebook's extraction cell (``extract_found_sessions``) runs the same step on
the same file and must extract the same frames.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
from moseq2_common.yaml import read_yaml, write_yaml

from moseq2_extract.gui import extract_found_sessions, generate_config_command

SESSION = "saline_example_0"
NUM_FRAMES = 100


def test_notebook_batch_extraction_honours_num_frames(tmp_path, raw_sessions, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(Path(raw_sessions) / SESSION, project / SESSION)
    monkeypatch.chdir(project)
    config_file = project / "config.yaml"
    generate_config_command(str(config_file), camera_type="k2")
    config = read_yaml(str(config_file))
    config["num_frames"] = NUM_FRAMES
    write_yaml(config, str(config_file))

    extract_found_sessions(str(project), str(config_file))

    with h5py.File(project / SESSION / "proc" / "results_00.h5", "r") as f:
        assert f["frames"].shape[0] == NUM_FRAMES
        assert f["timestamps"].shape[0] == NUM_FRAMES

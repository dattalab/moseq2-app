"""Extraction's file layer runs on the files it is given and knows nothing of projects.

``moseq2_extract.steps`` takes a recording, an output folder and an options
object. Run on a recording that sits inside a project folder, from that folder,
it still reads none of the project's files and writes nothing but its outputs:
the recording here sits in a trap project (``tests/layers.py``) whose files would
change the outputs if a step read them.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from fixtures import fetch_raw_sessions
from layers import listing, trap_project

from moseq2_common.options import read_settings
from moseq2_common.records import read_record
from moseq2_extract import steps
from moseq2_extract.options import EXTRACT_STAGE, RoiOptions
from moseq2_extract.setup import generate_config

SESSION = "saline_example_0"


def copy_recording(raw_sessions, folder: Path) -> Path:
    """A copy of the fixture recording ``SESSION`` in ``folder``/``SESSION``; its ``depth.avi``."""
    session = folder / SESSION
    session.mkdir(parents=True)
    for name in ("depth.avi", "depth_ts.txt", "metadata.json"):
        shutil.copy(Path(raw_sessions) / SESSION / name, session / name)
    return session / "depth.avi"


def kinect_options(folder: Path, cls):
    """``cls`` resolved from a Kinect ``config.yaml`` written in ``folder``, which is no project of the test's."""
    folder.mkdir(parents=True)
    config = folder / "config.yaml"
    generate_config(str(config), "kinect", overwrite=True)
    return cls.resolve(config=read_settings(config, cls, stage_classes=EXTRACT_STAGE))


def test_find_roi_on_plain_files_reads_and_writes_no_project_file(tmp_path, monkeypatch):
    opts = kinect_options(tmp_path / "elsewhere", RoiOptions)
    trap_folder = tmp_path / "trap"
    trap_folder.mkdir()
    recording = copy_recording(fetch_raw_sessions(), trap_folder)
    key = f"{SESSION}/depth.avi"
    trap = trap_project(trap_folder, entries={key: {"noise_tolerance": 1, "bg_roi_index": 3}},
                        sessions={key: "aggregate_results/listed_by_the_trap"})
    before = listing(trap_folder)
    monkeypatch.chdir(trap_folder)

    result = steps.find_roi(str(recording), None, opts)

    proc = recording.parent / "proc"
    assert result.roi_files == [proc / "roi_00.tiff"]
    assert (proc / "bground.tiff").is_file()
    record = read_record(proc / "roi_00.tiff")
    assert record["project"] is None
    assert [item["relative_path"] for item in record["inputs"]] == [None]
    assert [item["path"] for item in record["inputs"]] == [str(recording)]
    assert record["options"]["noise_tolerance"] == opts.noise_tolerance
    assert trap.unchanged()
    new = listing(trap_folder) - before
    assert new and all(path == proc or proc in path.parents for path in new), sorted(new)


def test_find_roi_takes_no_project():
    assert "project" not in inspect.signature(steps.find_roi).parameters
    assert "exclude" not in inspect.signature(steps.find_roi).parameters

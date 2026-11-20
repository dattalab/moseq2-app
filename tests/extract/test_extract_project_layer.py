"""Extraction's project layer gives each step its project's folders, settings files and record.

``moseq2_extract.project`` resolves where a recording's outputs go in the
project (its session's folder in ``aggregate_results/``), starts the run's record
with the project (so its inputs are described relative to the project folder and
include ``config.yaml`` and ``session_config.yaml``), and hands both to the file
layer. What it writes is what the step wrote when it took the project itself.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from fixtures import fetch_raw_sessions

from moseq2_common.options import read_settings
from moseq2_common.project import Project
from moseq2_common.records import read_record
from moseq2_common.yaml import write_yaml
from moseq2_extract import project as project_steps
from moseq2_extract.options import EXTRACT_STAGE, RoiOptions
from moseq2_extract.recordings import session_outputs
from moseq2_extract.session_config import recording_entry
from moseq2_extract.setup import generate_config

SESSION = "saline_example_0"
KEY = f"{SESSION}/depth.avi"


def call(project, recording, opts, **kwargs):
    """``find_roi`` in ``project``, through the project layer."""
    return project_steps.find_roi(project, recording, opts, **kwargs)


@pytest.fixture
def project(tmp_path):
    """A project holding one fixture recording, a Kinect config.yaml, a session_config.yaml entry for the
    recording, and a moseq2-project.yaml listing nothing yet."""
    folder = tmp_path / "proj"
    session = folder / SESSION
    session.mkdir(parents=True)
    for name in ("depth.avi", "depth_ts.txt", "metadata.json"):
        shutil.copy(Path(fetch_raw_sessions()) / SESSION / name, session / name)
    generate_config(str(folder / "config.yaml"), "kinect", overwrite=True)
    write_yaml({KEY: {"noise_tolerance": 25}}, str(folder / "session_config.yaml"))
    write_yaml({"sessions": {}, "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}},
                "applied_models": []}, str(folder / "moseq2-project.yaml"))
    return Project.at(folder)


def resolved(project, recording):
    """The arena options the command line resolves for ``recording`` in ``project``."""
    return RoiOptions.resolve(config=read_settings(project.config_file, RoiOptions, stage_classes=EXTRACT_STAGE),
                              entry=recording_entry(project, str(recording)))


def test_find_roi_in_a_project_writes_into_the_session_folder_and_records_the_project(project, monkeypatch):
    recording = project.folder / KEY
    opts = resolved(project, recording)
    assert opts.noise_tolerance == 25
    config_bytes = project.config_file.read_bytes()
    bookkeeping_bytes = project.bookkeeping_file.read_bytes()
    monkeypatch.chdir(project.folder)

    result = call(project, str(recording), opts)

    folder = session_outputs(project, KEY).folder
    assert folder == project.folder / "aggregate_results" / SESSION
    assert result.roi_files == [folder / "roi_00.tiff"]
    assert not (recording.parent / "proc").exists()
    record = read_record(folder / "roi_00.tiff")
    assert record["project"] == str(project.folder.resolve())
    assert [item["path"] for item in record["inputs"]] == [
        str(recording), str(project.config_file), str(project.session_config_file)]
    assert [item["relative_path"] for item in record["inputs"]] == [KEY, "config.yaml", "session_config.yaml"]
    assert record["options"]["noise_tolerance"] == 25
    assert "bg_roi_depth_range" in record["options"]
    assert project.config_file.read_bytes() == config_bytes
    assert project.bookkeeping_file.read_bytes() == bookkeeping_bytes


def test_find_roi_with_a_relative_config_file_writes_where_it_did(project, monkeypatch):
    """``find-roi session/depth.avi --config-file config.yaml`` from the project folder makes the project folder
    ``.``, so the session folder is the relative ``aggregate_results/<session>``, which ``get_roi_wrapper`` joins
    under the recording's folder. That is dattalab/moseq2-app#518; this pins where the images go until it is
    fixed."""
    import moseq2_extract.cli as extract_cli

    monkeypatch.chdir(project.folder)
    result = CliRunner().invoke(extract_cli.cli, ["find-roi", KEY, "--config-file", "config.yaml"],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output

    written = project.folder / SESSION / "aggregate_results" / SESSION
    assert (written / "roi_00.tiff").is_file()
    assert (written / "bground.tiff").is_file()
    assert not (project.folder / "aggregate_results").exists()

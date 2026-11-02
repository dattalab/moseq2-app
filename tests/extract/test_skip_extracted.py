"""Rerunning an extraction skips the recordings already extracted, quietly.

The notebook's extraction cell lists every recording in the project and, with
``skip_extracted = True`` (its default), leaves ``extract_wrapper`` to skip the
ones whose ``results_00.yaml`` says the extraction completed. A skipped
recording prints ``Skipping...`` and nothing else: the checks that guard an
extraction (the flip-classifier warning, the frame timestamps, the frame
count) are about a recording that is about to be extracted, and a finished
one is not. A project extracted before frame timestamps were required can hold
a finished session without a ``depth_ts.txt``; a rerun skips it like any other.

A recording not yet extracted still gets every check.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest
from moseq2_common.yaml import write_yaml

from moseq2_extract.gui import extract_found_sessions, generate_config_command
from moseq2_extract.helpers.wrappers import NoFlipClassifierWarning, extract_wrapper
from moseq2_extract.util import read_yaml

from fixtures import fetch_azure_mkv

SESSIONS = ("saline_example_0", "saline_example_1")
NO_TIMESTAMPS = "saline_example_1"


def mark_extracted(session):
    """What a finished extraction leaves behind, as far as skipping reads it."""
    (session / "proc").mkdir()
    write_yaml({"complete": True, "uuid": "x"}, str(session / "proc" / "results_00.yaml"))


@pytest.fixture
def extracted_project(tmp_path, raw_sessions, monkeypatch):
    """Two finished sessions; one was extracted without a depth_ts.txt."""
    project = tmp_path / "project"
    project.mkdir()
    for name in SESSIONS:
        shutil.copytree(Path(raw_sessions) / name, project / name)
        mark_extracted(project / name)
    (project / NO_TIMESTAMPS / "depth_ts.txt").unlink()
    monkeypatch.chdir(project)
    generate_config_command(str(project / "config.yaml"), camera_type="k2")
    return project


def rerun(project, capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_found_sessions(str(project), str(project / "config.yaml"), skip_extracted=True)
    captured = capsys.readouterr()
    return captured.out, captured.err, caught


def test_a_rerun_skips_finished_sessions_with_or_without_timestamps(extracted_project, capsys):
    out, err, caught = rerun(extracted_project, capsys)

    assert out.count("Skipping...") == len(SESSIONS), out
    assert "Extracted 0 of 0 recordings (2 already extracted and skipped)." in out, out
    assert "failed" not in out and "No frame timestamps" not in out, out
    assert out.rstrip().endswith("Extractions Complete."), out


def test_a_rerun_prints_only_skipping_for_each_session(extracted_project, capsys):
    out, err, caught = rerun(extracted_project, capsys)

    flip = [w for w in caught if issubclass(w.category, NoFlipClassifierWarning)]
    assert flip == [], [str(w.message) for w in flip]
    assert "without a flip classifier" not in out + err
    for name in SESSIONS:
        recording = extracted_project / name / "depth.avi"
        assert f"Processing: {recording}\nSkipping...\n" in out, out


# ---------------------------------------------------------------- single recordings


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate_config_command(str(tmp_path / "config.yaml"), camera_type="k2")
    return read_yaml(str(tmp_path / "config.yaml"))


@pytest.fixture
def unextracted(tmp_path, raw_sessions):
    """A copy of one fixture recording that has never been extracted."""
    session = tmp_path / "saline_example_0"
    shutil.copytree(Path(raw_sessions) / "saline_example_0", session)
    return session


def test_an_unextracted_recording_without_timestamps_still_fails(unextracted, config):
    (unextracted / "depth_ts.txt").unlink()
    with pytest.warns(NoFlipClassifierWarning), pytest.raises(FileNotFoundError, match="No frame timestamps"):
        extract_wrapper(str(unextracted / "depth.avi"), "proc", config, skip=True)


def test_an_unfinished_extraction_still_gets_the_frame_count_check(unextracted, config):
    """A results_00.yaml that is not complete does not count as extracted."""
    (unextracted / "proc").mkdir()
    write_yaml({"complete": False, "uuid": "x"}, str(unextracted / "proc" / "results_00.yaml"))
    lines = (unextracted / "depth_ts.txt").read_text().splitlines(keepends=True)
    (unextracted / "depth_ts.txt").write_text("".join(lines[:-1]))
    with pytest.warns(NoFlipClassifierWarning), pytest.raises(ValueError, match="Frame count mismatch"):
        extract_wrapper(str(unextracted / "depth.avi"), "proc", config, skip=True)


def test_a_finished_mkv_is_skipped_without_its_frame_count_line(tmp_path, config, capsys):
    session = tmp_path / "azure"
    session.mkdir()
    shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session / "depth.mkv")
    (session / "metadata.json").write_text("{}")
    mark_extracted(session)
    capsys.readouterr()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extract_wrapper(str(session / "depth.mkv"), "proc", config, skip=True)

    assert result is None
    out = capsys.readouterr().out
    assert out == f"Processing: {session / 'depth.mkv'}\nSkipping...\n"
    assert caught == [], [str(w.message) for w in caught]

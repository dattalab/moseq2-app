"""Rerunning an extraction skips the recordings already extracted, quietly.

A project's extraction (``moseq2 extract``, the notebook's Extract step) plans
first and leaves out every recording whose extraction is up to date: its
``results.h5`` is in its session folder with the record of an extraction made
with the settings and head/tail detection the project resolves now. When that is
every recording, the run extracts nothing and its summary says so. The checks
that guard an extraction (the flip-classifier warning, the frame timestamps,
the frame count) are about a recording that is about to be extracted, and a
finished one is not. A project extracted before frame timestamps were required
can hold a finished session without a ``depth_ts.txt``; a rerun leaves it out
like any other. ``extract_wrapper`` called on a finished recording with
``skip=True`` says ``Skipping...`` and nothing else.

A recording not yet extracted still gets every check.
"""

from __future__ import annotations

import shutil
import urllib.request
import warnings
from pathlib import Path

import pytest
from moseq2_common.yaml import read_yaml

from moseq2_extract import project as extraction
from moseq2_extract.gui import generate_config_command
from moseq2_extract.helpers.wrappers import NoFlipClassifierWarning, extract_wrapper

from extract_projects import CLASSIFIER, recorded_extraction, small_project
from fixtures import fetch_azure_mkv
from reporting import CollectingReporter

SESSIONS = ("saline_example_0", "saline_example_1")
NO_TIMESTAMPS = "saline_example_1"


def mark_extracted(session, folder=None):
    """What a finished extraction leaves behind, as far as skipping reads it: its ``results.h5``, in its
    session folder in the project's ``aggregate_results/`` (or in ``folder``)."""
    folder = folder if folder is not None else session.parent / "aggregate_results" / session.name
    folder.mkdir(parents=True)
    (folder / "results.h5").touch()


@pytest.fixture
def extracted_project(tmp_path, raw_sessions, monkeypatch):
    """Two finished sessions, each with the record of its extraction; one was extracted without a
    depth_ts.txt. config.yaml names no flip classifier: the project holds the published one already, so
    nothing is downloaded."""
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda *args, **kwargs: pytest.fail("the rerun downloaded head/tail detection"))
    project = small_project(tmp_path / "project", raw_sessions, SESSIONS, num_frames=None, classifier=False)
    shutil.copy(Path(raw_sessions) / CLASSIFIER, extraction.head_tail_detection(project).path)
    for name in SESSIONS:
        recorded_extraction(project, f"{name}/depth.avi")
    (Path(project.folder) / NO_TIMESTAMPS / "depth_ts.txt").unlink()
    monkeypatch.chdir(project.folder)
    return project


def rerun(project, capsys):
    """Run the project's extraction again; its run, what it reported, and the warnings and output it gave."""
    reporter = CollectingReporter()
    capsys.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = extraction.extract(project, reporter=reporter)
    captured = capsys.readouterr()
    return run, reporter, captured.out + captured.err, caught


def test_a_rerun_skips_finished_sessions_with_or_without_timestamps(extracted_project, capsys):
    run, reporter, printed, caught = rerun(extracted_project, capsys)

    assert (run.extracted, run.failed, run.jobs) == ([], {}, {})
    assert run.up_to_date == [f"{name}/depth.avi" for name in SESSIONS]
    assert reporter.statuses == ["Extracted 0 of 0 recordings."], reporter.statuses
    assert reporter.warnings == []
    assert printed == ""
    assert not any("failed" in line or "No frame timestamps" in line for line in reporter.statuses)


def test_a_rerun_extracts_nothing_and_warns_about_no_flip_classifier(extracted_project, capsys):
    run, reporter, printed, caught = rerun(extracted_project, capsys)

    flip = [w for w in caught if issubclass(w.category, NoFlipClassifierWarning)]
    assert flip == [], [str(w.message) for w in flip]
    assert not any("without a flip classifier" in text for text in reporter.statuses + reporter.warnings + [printed])
    assert reporter.bars == []
    assert not any("Processing:" in line for line in reporter.statuses)


# ---------------------------------------------------------------- single recordings


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate_config_command(str(tmp_path / "config.yaml"), camera_type="k2")
    data = read_yaml(str(tmp_path / "config.yaml"))
    # extraction's settings, as extract_wrapper takes them
    return {**data["extract"], "camera_type": data["camera_type"]}


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
    """An extraction left unfinished (its ``.h5`` under the temporary name) does not count as extracted."""
    (unextracted / "proc").mkdir()
    (unextracted / "proc" / ".results.h5.partial").touch()
    lines = (unextracted / "depth_ts.txt").read_text().splitlines(keepends=True)
    (unextracted / "depth_ts.txt").write_text("".join(lines[:-1]))
    with pytest.warns(NoFlipClassifierWarning), pytest.raises(ValueError, match="Frame count mismatch"):
        extract_wrapper(str(unextracted / "depth.avi"), "proc", config, skip=True)


def test_a_finished_mkv_is_skipped_without_its_frame_count_line(tmp_path, config, capsys):
    session = tmp_path / "azure"
    session.mkdir()
    shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session / "depth.mkv")
    (session / "metadata.json").write_text("{}")
    mark_extracted(session, session / "proc")
    capsys.readouterr()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extract_wrapper(str(session / "depth.mkv"), "proc", config, skip=True)

    assert result is None
    # with no reporter, status lines go to standard error
    err = capsys.readouterr().err
    assert err == f"Processing: {session / 'depth.mkv'}\nSkipping...\n"
    assert caught == [], [str(w.message) for w in caught]

"""The command line and the notebook extract the same recordings from a project.

``moseq2 extract`` and the notebook's Extract step run one step,
``moseq2_extract.project.extract``, which finds the project's recordings and
plans what to do with each. Both search every recording type the pipeline
reads (``.dat``, ``.mkv``, ``.avi``, ``.tar.gz``) unless told otherwise, and
both leave out a recording whose extraction is up to date: its ``results.h5``
is in its session folder in ``aggregate_results/`` with the record of an
extraction made as the project would make it now. An extraction carrying no
such record is redone by both.

Each test builds a project of empty recording files and stands in for the
extraction of each recording: the list it would have extracted is what it
found.

When every recording is up to date, both say in one line that they extracted
nothing; when only some are, they extract the rest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from moseq2_app.cli import cli
from extract_projects import recorded_extraction, small_project

#: One session folder per recording type.
RECORDINGS = {
    "dat_session": "depth.dat",
    "mkv_session": "depth.mkv",
    "avi_session": "depth.avi",
    "tar_session": "session_20260101.tar.gz",
}

#: The one line a run that extracts nothing ends with.
NOTHING = "Extracted 0 of 0 recordings."


@pytest.fixture
def project(tmp_path, raw_sessions, monkeypatch):
    """A project holding one recording of each type and the config.yaml generate-config writes, with the
    fixture's head/tail classifier."""
    project = small_project(tmp_path / "project", raw_sessions, (), num_frames=None)
    root = Path(project.folder)
    for folder, name in RECORDINGS.items():
        (root / folder).mkdir(parents=True)
        (root / folder / name).touch()
        (root / folder / "metadata.json").write_text("{}")
    monkeypatch.chdir(root)
    return project


@pytest.fixture
def extracted(monkeypatch):
    """The recordings the step extracts, in place of extracting them."""
    chosen = []
    monkeypatch.setattr(
        "moseq2_extract.project.extract_recording",
        lambda project, input_file, opts, **kw: chosen.append(str(input_file)),
    )
    return chosen


def mark_extracted(project, folder, *, record=True):
    """What a finished extraction leaves behind, as far as the plan reads it: its session's ``results.h5``,
    with the record of an extraction made with the project's settings, or without any (``record=False``)."""
    recorded_extraction(project, f"{folder}/{RECORDINGS[folder]}", record=record)


def every_recording(root):
    return sorted(str(Path(root) / folder / name) for folder, name in RECORDINGS.items())


def run_cli(project, *flags):
    result = CliRunner().invoke(cli, ["extract", *flags])
    assert result.exit_code == 0, result.output
    return result


def run_notebook(project):
    """The notebook's Extract step, run in the project folder (the working folder); its widget."""
    from moseq2_app.notebook import extract

    assert Path(os.getcwd()) == Path(project.folder)
    widget = extract()
    widget.start()
    run = widget.wait()
    assert run["state"] == "finished", widget.problem
    return widget


def test_batch_extract_finds_every_recording_type_by_default(project, extracted):
    run_cli(project)

    assert sorted(extracted) == every_recording(project.folder)


@pytest.mark.parametrize(
    "record",
    [True, False],
    ids=["finished-left-out", "finished-without-record-included"],
)
def test_both_front_ends_list_the_same_recordings(project, extracted, record):
    mark_extracted(project, "avi_session", record=record)
    finished = str(Path(project.folder) / "avi_session" / "depth.avi")
    expected = [r for r in every_recording(project.folder) if not (record and r == finished)]

    run_cli(project)
    by_cli = sorted(extracted)
    extracted.clear()
    run_notebook(project)
    by_notebook = sorted(extracted)

    assert by_cli == expected
    assert by_notebook == expected


def test_a_rerun_with_every_recording_extracted_says_so_in_one_line(project, extracted):
    for folder in RECORDINGS:
        mark_extracted(project, folder)

    by_cli = run_cli(project).stderr
    widget = run_notebook(project)
    by_notebook = [line["text"] for line in widget.lines]

    assert extracted == []
    assert by_cli == NOTHING + "\n"
    assert by_notebook == [NOTHING]


def test_a_rerun_with_some_recordings_extracted_extracts_the_rest(project, extracted):
    mark_extracted(project, "avi_session")

    output = run_cli(project).stderr

    assert sorted(extracted) == [r for r in every_recording(project.folder) if not r.endswith("depth.avi")]
    assert NOTHING not in output, output
    assert output.endswith("Extracted 3 of 3 recordings.\n"), output

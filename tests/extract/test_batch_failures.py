"""A run of extraction ends by saying which recordings failed.

``moseq2_extract.project.extract``, which ``moseq2 extract`` and the notebook's
Extract step run, extracts one recording at a time on this computer and records
each recording's error so the rest still run. With many recordings an error in
the middle scrolls past, so the run ends with a summary: how many of the
recordings it set out to extract were extracted, and each one that failed with
its error. A recording the plan found up to date is left as it is and counted in
neither.

The Extract step shows each failed recording's row as failed and the summary
among its lines; ``moseq2 extract`` exits with a non-zero status: 1 when every
failure was an expected one, 3 when one looks like a bug.

The extraction itself is replaced by a stand-in that fails for one recording.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import moseq2_extract.steps as extract_steps
from moseq2_app.cli import cli
from moseq2_common.errors import ExpectedError
from moseq2_common.yaml import read_yaml, write_config
from moseq2_extract import project as extraction

from extract_projects import recorded_extraction, small_project
from reporting import CollectingReporter

SESSIONS = ("session_a", "session_b", "session_c", "session_d")
FAILS = "session_b"
UP_TO_DATE = "session_d"


def key(name):
    return f"{name}/depth.avi"


@pytest.fixture
def project(tmp_path, raw_sessions):
    """Four recordings in a project extracting locally, with the fixture's head/tail classifier; one of
    them is extracted already, with the record of an extraction made with the project's settings."""
    project = small_project(tmp_path / "project", raw_sessions, (), num_frames=None)
    for name in SESSIONS:
        session = Path(project.folder) / name
        session.mkdir()
        (session / "depth.avi").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
        (session / "metadata.json").write_text("{}")
    data = read_yaml(str(project.config_file))
    data["computing"]["extract"]["cluster_type"] = "local"
    write_config(data, str(project.config_file))
    recorded_extraction(project, key(UP_TO_DATE))
    return project


class BrokenRecording(ExpectedError, ValueError):
    """A recording the stand-in cannot extract, as an extraction step reports a problem with the data."""


def stand_in(fails=(FAILS,), error=ValueError):
    """An extraction that fails for some recordings with ``error``.

    ``extract_wrapper`` returns the output directory when it extracts a recording.
    """

    def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False, uuid=None, record=None):
        if any(name in input_file for name in fails):
            raise error(f"the depth video in {input_file} is broken\nsecond line")
        return output_dir

    return extract_wrapper


def recording(project, name):
    return str(Path(project.folder) / name / "depth.avi")


def test_the_batch_ends_with_a_summary_of_its_failures(project, monkeypatch):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in())
    reporter = CollectingReporter()

    run = extraction.extract(project, reporter=reporter)

    failed = recording(project, FAILS)
    assert reporter.statuses[-1] == (
        "Extracted 2 of 3 recordings; 1 failed.\n"
        "Failed:\n"
        f"  {failed}\n"
        f"    ValueError: the depth video in {failed} is broken\n"
        "    second line\n"
        "Fix the problem each error names, then run the extraction again with `moseq2 extract` (or the "
        "notebook's Extract step): the recordings extracted already are left as they are."
    )
    assert list(run.failed) == [key(FAILS)]
    assert sorted(run.extracted) == [key(name) for name in SESSIONS if name not in (FAILS, UP_TO_DATE)]
    assert run.up_to_date == [key(UP_TO_DATE)]


def test_a_batch_without_failures_says_how_many_were_extracted(project, monkeypatch):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in(fails=()))
    reporter = CollectingReporter()

    run = extraction.extract(project, reporter=reporter)

    assert run.failed == {}
    assert reporter.statuses[-1] == "Extracted 3 of 3 recordings."
    assert not any("Failed" in line for line in reporter.statuses)


def notebook_run(project):
    """Run the notebook's Extract step in ``project``'s folder; the widget and its finished run."""
    from moseq2_app.notebook import extract

    before = os.getcwd()
    os.chdir(project.folder)
    try:
        widget = extract()
        widget.start()
        run = widget.wait()
    finally:
        os.chdir(before)
    return widget, run


def test_the_notebook_reports_failures_instead_of_complete(project, monkeypatch):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in())

    widget, run = notebook_run(project)

    states = {item: value["state"] for item, value in run["items"].items()}
    assert states == {key(name): "failed" if name == FAILS else "done"
                      for name in SESSIONS if name != UP_TO_DATE}
    statuses = [line["text"] for line in widget.lines if line["kind"] == "status"]
    assert statuses[-1].startswith("Extracted 2 of 3 recordings; 1 failed")
    assert not any("Extractions Complete." in text for text in statuses)


def test_the_notebook_says_complete_when_nothing_failed(project, monkeypatch):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in(fails=()))

    widget, run = notebook_run(project)

    assert run["state"] == "finished", widget.problem
    assert {value["state"] for value in run["items"].values()} == {"done"}
    assert len(run["items"]) == 3
    statuses = [line["text"] for line in widget.lines if line["kind"] == "status"]
    assert statuses[-1] == "Extracted 3 of 3 recordings."


def moseq2_extract(project):
    before = os.getcwd()
    os.chdir(project.folder)
    try:
        return CliRunner().invoke(cli, ["extract"])
    finally:
        os.chdir(before)


@pytest.mark.parametrize("fails, error, status", [((FAILS,), BrokenRecording, 1), ((FAILS,), ValueError, 3),
                                                  ((), ValueError, 0)],
                         ids=["one-fails", "one-fails-as-a-bug", "none-fail"])
def test_batch_extract_exits_non_zero_when_a_recording_failed(project, monkeypatch, fails, error, status):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in(fails=fails, error=error))

    result = moseq2_extract(project)

    assert result.exit_code == status, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception


def test_batch_extract_without_a_config_file_stops_and_says_what_to_do(tmp_path, monkeypatch):
    """Outside a project, ``moseq2 extract`` stops with a non-zero status, naming the folder it looked in."""
    recordings = tmp_path / "recordings"
    (recordings / "session_a").mkdir(parents=True)
    (recordings / "session_a" / "depth.avi").write_bytes(b"\0")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["extract"])
    assert result.exit_code == 1
    assert result.stderr == (
        f"Error: `moseq2 extract` runs in a project folder, and {tmp_path} has no config.yaml.\n"
        "To fix it: run it from your project folder, or start a project here with `moseq2 setup`.\n"
    )

"""The notebook's Extract step: ``from moseq2_app.notebook import extract`` then ``extract()``.

Running the cell only opens the GUI. Its button starts extraction in the
background, saving the settings changed on the page first; each recording's row
follows its extraction; a problem is a panel saying what happened and the fix,
or, for a bug, that something unexpected went wrong with the traceback folded
away. Stop ends a local run at the next chunk, leaving what was extracted before
as it was, and nothing a run prints reaches the cell.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path

import pytest

from moseq2_common.computing import available_cores
from moseq2_common.project import step_lock
from moseq2_common.yaml import read_yaml, write_config
from extract_projects import small_project

TWO = ("saline_example_0", "saline_example_1")


@pytest.fixture
def in_folder():
    """Run a block with the working folder set, as a notebook in that folder runs."""
    before = os.getcwd()

    def chdir(folder):
        os.chdir(folder)

    yield chdir
    os.chdir(before)


def _open(project, in_folder):
    from moseq2_app.notebook import extract

    in_folder(project.folder)
    return extract()


def test_running_the_cell_only_opens_the_gui(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    widget = _open(project, in_folder)

    assert widget.run["state"] == "idle"
    assert [(r["id"], r["label"], r["plan"]) for r in widget.recordings] == [(f"{n}/depth.avi", n, "new") for n in TWO]
    assert [s["title"] for s in widget.sections] == ["Where it runs", "More settings"]
    assert widget.problem == {}
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_start_runs_in_the_background_and_saves_the_settings(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1], num_frames=30)
    widget = _open(project, in_folder)

    widget.start({**widget.saved, "max_height": 110})
    assert widget.run["state"] == "running"
    run = widget.wait(300)

    assert run["state"] == "finished", widget.problem
    assert run["items"][f"{TWO[0]}/depth.avi"]["state"] == "done", run
    assert read_yaml(str(project.config_file))["extract"]["max_height"] == 110
    assert "cores" not in read_yaml(str(project.config_file))["computing"]["extract"] or \
        read_yaml(str(project.config_file))["computing"]["extract"]["cores"] is None
    assert (Path(project.folder) / "aggregate_results" / TWO[0] / "results.h5").is_file()
    assert [r["plan"] for r in widget.recordings] == ["done"]


def test_rows_follow_the_run(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1], num_frames=60, extract={"chunk_size": 20})
    widget = _open(project, in_folder)
    seen = []
    widget.observe(lambda change: seen.append(dict(change["new"]["items"].get(f"{TWO[0]}/depth.avi", {}))),
                   names="run")

    widget.start()
    run = widget.wait(300)

    assert run["state"] == "finished", widget.problem
    progress = [(item.get("state"), item.get("done"), item.get("total")) for item in seen if item]
    assert ("running", 0, 3) in progress
    assert ("running", 1, 3) in progress and ("running", 2, 3) in progress
    assert progress[-1] == ("done", 3, 3)


def test_an_expected_error_is_a_problem_panel_with_its_fix(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1])
    widget = _open(project, in_folder)

    with step_lock(project, "extract", where="command line"):
        widget.start()
        run = widget.wait(60)

    assert run["state"] == "failed"
    problem = widget.problem
    assert problem["kind"] == "expected"
    assert problem["title"].startswith("Extract is already running in this project: it was started at ")
    assert " on the command line (process " in problem["title"]
    assert problem["fix"] == "wait for it to finish, or stop it where it was started."


def test_a_bug_is_a_panel_with_the_traceback_folded(tmp_path, raw_sessions, in_folder, monkeypatch):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1])
    widget = _open(project, in_folder)

    def broken(*args, **kwargs):
        print("printed before the bug")
        raise KeyError("a bug")

    monkeypatch.setattr("moseq2_extract.project.extract", broken)
    widget.start()
    run = widget.wait(60)

    assert run["state"] == "failed"
    problem = widget.problem
    assert (problem["kind"], problem["title"]) == ("bug", "Something unexpected went wrong")
    assert problem["text"] == ("This looks like a bug in MoSeq2, not a problem with your data or settings. Please "
                               "report it at https://github.com/dattalab/moseq2-app/issues and include the details "
                               "below.")
    assert "Traceback (most recent call last)" in problem["details"]
    assert "KeyError: 'a bug'" in problem["details"]
    assert "printed before the bug" in problem["details"]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_stop_ends_a_local_run_at_the_next_chunk(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1], num_frames=60, extract={"chunk_size": 20})
    widget = _open(project, in_folder)
    widget.start()
    assert widget.wait(300)["state"] == "finished", widget.problem
    h5 = Path(project.folder) / "aggregate_results" / TWO[0] / "results.h5"
    before = _digest(h5)

    data = read_yaml(str(project.config_file))
    data["extract"]["max_height"] = 100
    write_config(data, str(project.config_file))
    widget.refresh()
    assert [r["plan"] for r in widget.recordings] == ["redo"]

    def stop_after_first_chunk(change):
        item = change["new"]["items"].get(f"{TWO[0]}/depth.avi", {})
        if item.get("done", 0) >= 1:
            widget.stop()

    widget.observe(stop_after_first_chunk, names="run")
    widget.start()
    run = widget.wait(300)

    assert run["state"] == "stopped"
    assert widget.problem["title"].startswith("Stopped at ")
    assert widget.problem["title"].endswith(". Recordings that hadn't finished keep what they had before.")
    assert widget.problem["fix"] == "press Extract to finish them."
    assert _digest(h5) == before
    assert not list(h5.parent.glob(".*partial*"))


def test_nothing_a_run_prints_reaches_the_cell(tmp_path, raw_sessions, in_folder, monkeypatch):
    import IPython.display

    project = small_project(tmp_path / "project", raw_sessions, TWO[:1], num_frames=30)
    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda *objects, **kwargs: shown.append(objects))
    cell_out, cell_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", cell_out)
    monkeypatch.setattr(sys, "stderr", cell_err)

    widget = _open(project, in_folder)
    widget.start()
    run = widget.wait(300)

    assert run["state"] == "finished", widget.problem
    assert (cell_out.getvalue(), cell_err.getvalue()) == ("", "")
    assert shown == []


def test_cores_start_at_every_core_this_computer_may_use(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1])

    widget = _open(project, in_folder)

    assert widget.saved["cores"] == available_cores()
    (cores,) = [f for s in widget.sections for f in s["fields"] if f["key"] == "cores"]
    assert (cores["min"], cores["max"], cores["slider"]) == (1, available_cores(), True)


def test_rows_name_each_recording_by_its_session():
    from moseq2_app.notebook._widget import session_labels

    assert session_labels(["saline_example_0/depth.avi", "cohort1/saline_example_1/depth.avi"]) == {
        "saline_example_0/depth.avi": "saline_example_0", "cohort1/saline_example_1/depth.avi": "saline_example_1"}
    # two folders of one name: the folder above tells them apart
    assert session_labels(["a/mouse/depth.avi", "b/mouse/depth.avi"]) == {
        "a/mouse/depth.avi": "a/mouse", "b/mouse/depth.avi": "b/mouse"}
    # two recordings in one folder keep their files; one in the project folder is its file
    assert session_labels(["day1/one.avi", "day1/two.avi", "depth.dat"]) == {
        "day1/one.avi": "day1/one.avi", "day1/two.avi": "day1/two.avi", "depth.dat": "depth.dat"}


def test_the_gui_help_names_no_command_line_values(tmp_path, raw_sessions, in_folder):
    project = small_project(tmp_path / "project", raw_sessions, TWO[:1])

    widget = _open(project, in_folder)

    helps = {f["key"]: f["help"] for s in widget.sections for f in s["fields"]}
    assert helps["cluster_type"] == "Where the work runs: on this computer, or as jobs on a SLURM cluster."
    for key, text in helps.items():
        for value in ("(local)", "(slurm)", "16GB", "4:00:00", "auto"):
            assert value not in text, (key, text)

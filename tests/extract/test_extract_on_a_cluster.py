"""Extract on a SLURM cluster: one job per recording, followed from any notebook or terminal until it ends.

With ``computing.extract.cluster_type: slurm``, ``moseq2_extract.project.extract``
submits one job per recording it extracts (``sbatch --parsable``) and records the
job's id beside the recording's outputs (``extract-submitted.yaml``). Each job
reports through a ``ProgressFileReporter`` into ``extract-progress.json``;
``cluster_runs`` reads those files and one ``squeue`` (then one ``sacct`` for the
jobs gone from the queue) into each recording's state, falling back to the
progress files where the cluster keeps no accounting. The Extract GUI polls while
it shows a cluster run, Stop cancels the jobs (``stop``), ``moseq2 extract --wait``
follows them (``follow``), a GUI opened later picks them up again, and a rerun
submits nothing for a recording whose job is queued or running.

Every test runs against the stand-ins in ``tests/fake_slurm`` (``slurm_fakes``);
no SLURM command runs.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml, write_config
from extract_projects import small_project
from reporting import CollectingReporter
from slurm_fakes import fake_slurm  # noqa: F401 -- the fixture

TWO = ("saline_example_0", "saline_example_1")
KEYS = tuple(f"{name}/depth.avi" for name in TWO)

#: What W14 asks for, with the job's log named.
LOG_FIX = "read the job's log, {log}, then press Extract again (`moseq2 extract` on the command line)."


def _project(tmp_path, raw_sessions, names=TWO, **computing):
    project = small_project(tmp_path / "project", raw_sessions, names, num_frames=30)
    data = read_yaml(str(project.config_file))
    data["computing"]["extract"].update(cluster_type="slurm", **computing)
    write_config(data, str(project.config_file))
    return project


def _folder(project, name) -> Path:
    return Path(project.folder) / "aggregate_results" / name


def _submitted(project, name) -> dict:
    return read_yaml(str(_folder(project, name) / "extract-submitted.yaml"))


def _now(seconds_ago: float = 0) -> str:
    return (datetime.now().astimezone() - timedelta(seconds=seconds_ago)).isoformat(timespec="seconds")


def _progress(project, name, *, job, state="running", done=0, total=None, seconds_ago=0, message=None, fix=None):
    """Write ``name``'s progress file as its job writes it."""
    path = _folder(project, name) / "extract-progress.json"
    path.write_text(json.dumps({"state": state, "stage": "Processing batches", "done": done, "total": total,
                                "updated": _now(seconds_ago), "message": message, "fix": fix, "job": job}))
    return path


def _run(args, cwd):
    from moseq2_app.cli import cli

    before = os.getcwd()
    os.chdir(cwd)
    try:
        return CliRunner().invoke(cli, [str(a) for a in args], prog_name="moseq2")
    finally:
        os.chdir(before)


def _extract(project):
    from moseq2_extract import project as extraction

    return extraction.extract(project, reporter=CollectingReporter())


def _open(project, monkeypatch):
    from moseq2_app.notebook import extract

    monkeypatch.chdir(project.folder)
    return extract()


def _states(widget) -> dict:
    return {key: (item["state"], item.get("done"), item.get("total")) for key, item in widget.run["items"].items()}


# ------------------------------------------------------------------------------- submitting


def test_submitting_records_each_recordings_job_id_beside_its_record(tmp_path, raw_sessions, fake_slurm):
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)

    run = _extract(project)

    assert sorted(run.jobs) == list(KEYS)
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 2
    for number, (name, call) in enumerate(zip(TWO, calls)):
        folder = _folder(project, name)
        submitted = _submitted(project, name)
        assert str(submitted["job_id"]) == str(1001 + number)
        assert (submitted["settings_file"], submitted["log"]) == ("extract-job.yaml", "extract-job.log")
        assert datetime.fromisoformat(submitted["submitted"]).tzinfo is not None
        words = shlex.split(submitted["sbatch"])
        assert words[1:] == call, "the line recorded is the line run"
        assert words[:6] == ["sbatch", "--parsable", "-J", "moseq2-extract", "-o", str(folder / "extract-job.log")]
        assert words[6:14] == ["-c", "4", "--mem=16GB", "-p", "short", "-t", "1:00:00", "--wrap"]
        assert words[14] == shlex.join([sys.executable, "-m", "moseq2_app", "run-job",
                                        str(folder / "extract-job.yaml")]) + "; "
        assert (folder / "extract-job.yaml").is_file()
    assert not (Path(project.folder) / "extract_out.sh").exists()
    assert not list(Path(project.folder).rglob("results.h5"))


def test_moseq2_extract_submits_and_says_how_to_follow(tmp_path, raw_sessions, fake_slurm):
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)

    result = _run(["extract"], project.folder)

    assert result.exit_code == 0, result.output
    assert ("Submitted 2 jobs (1001 to 1002). Run `moseq2 extract` again to see where they stand, "
            "`moseq2 extract --wait` to follow them, or `moseq2 extract --stop` to cancel them.") in result.stderr

    fake_slurm.set("1001", state="RUNNING")
    _progress(project, TWO[0], job="1001", done=1, total=2)
    again = _run(["extract"], project.folder)

    assert again.exit_code == 0, again.output
    assert f"{KEYS[0]}: in progress on the cluster, running as job 1001 (50% done)." in again.stderr
    assert f"{KEYS[1]}: in progress on the cluster, waiting in the queue as job 1002." in again.stderr
    assert len(fake_slurm.calls("sbatch")) == 2, "a rerun submitted a recording whose job is on the cluster"


def test_sbatch_refusing_a_job_names_its_answer(tmp_path, raw_sessions, fake_slurm):
    from moseq2_common.slurm import SubmitError

    fake_slurm.hold()
    fake_slurm.fail("REFUSED", TWO[1])
    project = _project(tmp_path, raw_sessions)

    with pytest.raises(SubmitError) as raised:
        _extract(project)

    error = raised.value
    assert error.message == (f"The cluster refused the job for {KEYS[1]}. sbatch said: sbatch: error: Batch job "
                             "submission failed: Invalid partition name specified")
    assert error.fix == ("check that the partition short exists and that this computer can submit jobs "
                         "(`sbatch --version`), then press Extract again.")
    assert str(_submitted(project, TWO[0])["job_id"]) == "1001", "the job submitted before stays recorded"
    assert not (_folder(project, TWO[1]) / "extract-submitted.yaml").exists()

    result = _run(["extract"], project.folder)
    assert result.exit_code == 1
    assert f"Error: The cluster refused the job for {KEYS[1]}." in result.stderr
    assert "To fix it: check that the partition short exists" in result.stderr


# ------------------------------------------------------------------------------- following


def test_rows_show_waiting_running_done_from_the_queue_and_progress_files(tmp_path, raw_sessions, fake_slurm,
                                                                         monkeypatch):
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    widget = _open(project, monkeypatch)

    widget.start()
    run = widget.wait(120)

    assert (run["state"], run["where"]) == ("running", "the cluster"), widget.problem
    assert _states(widget) == {KEYS[0]: ("queued", 0, None), KEYS[1]: ("queued", 0, None)}
    assert [r["plan"] for r in widget.recordings] == ["in progress", "in progress"]

    fake_slurm.set("1001", state="RUNNING")
    _progress(project, TWO[0], job="1001", done=2, total=5)
    polls = len(fake_slurm.calls("squeue"))
    widget.poll()
    assert len(fake_slurm.calls("squeue")) == polls + 1, "one squeue per poll"
    assert _states(widget) == {KEYS[0]: ("running", 2, 5), KEYS[1]: ("queued", 0, None)}

    fake_slurm.set("1001", state="COMPLETED")
    _progress(project, TWO[0], job="1001", state="finished", done=5, total=5)
    fake_slurm.set("1002", state="RUNNING")
    widget.poll()
    assert _states(widget) == {KEYS[0]: ("done", 5, 5), KEYS[1]: ("running", 0, None)}
    assert widget.run["state"] == "running"

    fake_slurm.set("1002", state="COMPLETED")
    _progress(project, TWO[1], job="1002", state="finished", done=4, total=4)
    widget.poll()
    assert widget.run["state"] == "finished"
    assert widget.run["finished_at"]
    assert _states(widget) == {KEYS[0]: ("done", 5, 5), KEYS[1]: ("done", 4, 4)}


def test_the_job_reports_through_the_same_reporter_interface(tmp_path, raw_sessions, fake_slurm, monkeypatch):
    from moseq2_common.report import ProgressFileReporter, Reporter, read_progress
    from moseq2_extract import project as extraction

    assert isinstance(ProgressFileReporter(tmp_path / "p.json"), Reporter)
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    _extract(project)
    monkeypatch.setenv("SLURM_JOB_ID", "1001")

    extraction.run_job(_folder(project, TWO[0]) / "extract-job.yaml", reporter=CollectingReporter())

    progress = read_progress(_folder(project, TWO[0]) / "extract-progress.json")
    assert set(progress) >= {"state", "stage", "done", "total", "updated", "message"}
    assert progress["state"] == "finished"
    assert progress["total"] and progress["done"] == progress["total"]
    assert progress["message"] is None and progress["job"] == "1001"
    assert datetime.fromisoformat(progress["updated"]).tzinfo is not None
    assert (_folder(project, TWO[0]) / "results.h5").is_file()

    # a job that stops on a problem with its recording names the problem
    monkeypatch.setenv("SLURM_JOB_ID", "1002")
    recording = Path(project.folder) / KEYS[1]
    recording.unlink()
    with pytest.raises(Exception) as raised:
        extraction.run_job(_folder(project, TWO[1]) / "extract-job.yaml", reporter=CollectingReporter())
    progress = read_progress(_folder(project, TWO[1]) / "extract-progress.json")
    assert progress["state"] == "failed"
    assert progress["message"] == str(raised.value)
    assert progress["job"] == "1002"


def test_wait_follows_the_jobs_until_they_finish(tmp_path, raw_sessions, fake_slurm):
    project = _project(tmp_path, raw_sessions)

    result = _run(["extract", "--wait"], project.folder)

    assert result.exit_code == 0, result.output
    for key, name in zip(KEYS, TWO):
        assert (_folder(project, name) / "results.h5").is_file()
        assert f"{key}: extracted." in result.stderr
        assert (_folder(project, name) / "extract-job.log").is_file()
    assert "Extracted 2 of 2 recordings." in result.stderr
    assert set(job["state"] for job in fake_slurm.jobs().values()) == {"COMPLETED"}

    from moseq2_extract import project as extraction

    assert [p.plan for p in extraction.plan(project)] == ["done", "done"]


# ------------------------------------------------------------------------------- failures


def test_out_of_memory_becomes_the_problem_with_a_bigger_memory_fix(tmp_path, raw_sessions, fake_slurm,
                                                                   monkeypatch):
    from moseq2_extract import project as extraction

    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    widget = _open(project, monkeypatch)
    widget.start()
    widget.wait(120)

    fake_slurm.set("1001", state="OUT_OF_MEMORY", exit_code="0:125")
    run = extraction.cluster_runs(project)[KEYS[0]]

    assert run.state == "failed"
    assert run.problem.message == "The job ran out of its 16 GB of memory."
    assert run.problem.fix == "raise Memory per job to 24 GB and press Extract again."
    assert run.problem.command_line_fix == "run `moseq2 extract --memory 24GB`."

    widget.poll()
    item = widget.run["items"][KEYS[0]]
    assert item["state"] == "failed"
    assert (item["problem"]["title"], item["problem"]["fix"]) == (
        "The job ran out of its 16 GB of memory.", "raise Memory per job to 24 GB and press Extract again.")


def test_out_of_time_suggests_twice_the_time(tmp_path, raw_sessions, fake_slurm):
    from moseq2_extract import project as extraction

    fake_slurm.fail("TIMEOUT", TWO[0])
    project = _project(tmp_path, raw_sessions, TWO[:1])
    _extract(project)
    fake_slurm.wait_for(lambda jobs: jobs["1001"]["state"] == "TIMEOUT")

    run = extraction.cluster_runs(project)[KEYS[0]]

    assert run.state == "failed"
    assert run.problem.message == "The job ran out of its 1 hour."
    assert run.problem.fix == "raise Time per job to 2 hours and press Extract again."
    assert run.problem.command_line_fix == "run `moseq2 extract --wall-time 2:00:00`."


def test_a_job_that_stops_on_a_problem_shows_the_problem_it_found(tmp_path, raw_sessions, fake_slurm):
    from moseq2_extract import project as extraction

    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    _extract(project)
    fake_slurm.set("1001", state="FAILED", exit_code="1:0")
    _progress(project, TWO[0], job="1001", state="failed", message="The recording has no timestamps.",
              fix="put depth_ts.txt beside it.")
    fake_slurm.set("1002", state="NODE_FAIL", reason="NodeDown")

    runs = extraction.cluster_runs(project)

    assert (runs[KEYS[0]].problem.message, runs[KEYS[0]].problem.fix) == (
        "The recording has no timestamps.", "put depth_ts.txt beside it.")
    assert runs[KEYS[1]].problem.message == "The job stopped with SLURM state NODE_FAIL (NodeDown)."
    assert runs[KEYS[1]].problem.fix == LOG_FIX.format(log=_folder(project, TWO[1]) / "extract-job.log")


def test_a_finished_job_is_judged_from_its_progress_file_when_sacct_is_missing(tmp_path, raw_sessions,
                                                                              fake_slurm):
    from moseq2_extract import project as extraction

    fake_slurm.hold()
    fake_slurm.no_accounting()
    project = _project(tmp_path, raw_sessions)
    _extract(project)
    fake_slurm.set("1001", state="COMPLETED")
    _progress(project, TWO[0], job="1001", state="finished", done=3, total=3)

    runs = extraction.cluster_runs(project)

    assert (runs[KEYS[0]].state, runs[KEYS[0]].done, runs[KEYS[0]].total) == ("done", 3, 3)
    assert runs[KEYS[1]].state == "queued"

    # and where sacct is not installed at all
    (fake_slurm.bin / "sacct").unlink()
    assert extraction.cluster_runs(project)[KEYS[0]].state == "done"


def test_a_job_gone_from_the_queue_with_a_stalled_progress_file_is_failed(tmp_path, raw_sessions, fake_slurm):
    from moseq2_extract import project as extraction

    fake_slurm.hold()
    fake_slurm.no_accounting()
    project = _project(tmp_path, raw_sessions)
    _extract(project)
    for job in ("1001", "1002"):
        fake_slurm.set(job, state="COMPLETED")
    _progress(project, TWO[0], job="1001", done=2, total=5, seconds_ago=extraction.STALLED_AFTER + 60)
    _progress(project, TWO[1], job="1002", done=2, total=5, seconds_ago=10)

    runs = extraction.cluster_runs(project)

    assert runs[KEYS[0]].state == "failed"
    assert runs[KEYS[0]].problem.message == ("The job left the queue without finishing, and the cluster keeps no "
                                             "record of why.")
    assert runs[KEYS[0]].problem.fix == LOG_FIX.format(log=_folder(project, TWO[0]) / "extract-job.log")
    assert runs[KEYS[1]].state == "running", "a progress file written moments ago is a job still writing"


def test_progress_reader_never_moves_a_bar_backwards_or_fails_on_a_half_written_file(tmp_path, raw_sessions,
                                                                                    fake_slurm):
    from moseq2_common.report import read_progress
    from moseq2_extract import project as extraction

    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions, TWO[:1])
    _extract(project)
    fake_slurm.set("1001", state="RUNNING")
    path = _progress(project, TWO[0], job="1001", done=5, total=10, seconds_ago=0)
    runs = extraction.cluster_runs(project)
    assert (runs[KEYS[0]].state, runs[KEYS[0]].done) == ("running", 5)

    _progress(project, TWO[0], job="1001", done=3, total=10, seconds_ago=30)  # an older copy, from a cache
    runs = extraction.cluster_runs(project, previous=runs)
    assert (runs[KEYS[0]].state, runs[KEYS[0]].done, runs[KEYS[0]].total) == ("running", 5, 10)

    path.write_text("")
    assert read_progress(path) is None
    runs = extraction.cluster_runs(project, previous=runs)
    assert (runs[KEYS[0]].state, runs[KEYS[0]].done) == ("running", 5)

    path.write_text('{"state": "running", "do')
    assert read_progress(path) is None
    runs = extraction.cluster_runs(project, previous=runs)
    assert (runs[KEYS[0]].state, runs[KEYS[0]].done) == ("running", 5)

    path.unlink()
    assert read_progress(path) is None
    runs = extraction.cluster_runs(project, previous=runs)
    assert (runs[KEYS[0]].state, runs[KEYS[0]].done) == ("running", 5)


# ------------------------------------------------------------------------------- stopping and coming back


def test_stop_cancels_the_recorded_jobs(tmp_path, raw_sessions, fake_slurm, monkeypatch):
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    widget = _open(project, monkeypatch)
    widget.start()
    widget.wait(120)
    fake_slurm.set("1001", state="RUNNING")

    widget.stop()

    assert fake_slurm.calls("scancel") == [["1001", "1002"]]
    assert {job["state"] for job in fake_slurm.jobs().values()} == {"CANCELLED"}
    assert widget.run["state"] == "stopped"
    assert {item["state"] for item in widget.run["items"].values()} == {"cancelled"}
    assert widget.problem["title"].startswith("Cancelled 2 cluster jobs at ")
    assert widget.problem["fix"] == "press Extract to submit the recordings that hadn't finished."
    for name in TWO:
        assert _submitted(project, name)["cancelled"]

    # a cancelled recording is extracted again by the next run; `moseq2 extract --stop` cancels from a terminal
    assert _run(["extract"], project.folder).exit_code == 0
    result = _run(["extract", "--stop"], project.folder)
    assert result.exit_code == 0, result.output
    assert "Cancelled 2 cluster jobs (1003, 1004)." in result.stderr
    assert fake_slurm.calls("scancel")[-1] == ["1003", "1004"]


def test_reopening_picks_the_jobs_up_again(tmp_path, raw_sessions, fake_slurm, monkeypatch):
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    first = _open(project, monkeypatch)
    first.start()
    first.wait(120)
    fake_slurm.set("1001", state="RUNNING")
    _progress(project, TWO[0], job="1001", done=3, total=5)
    first.poll()

    reopened = _open(project, monkeypatch)

    assert (reopened.run["state"], reopened.run["where"]) == ("running", "the cluster")
    assert _states(reopened) == _states(first) == {KEYS[0]: ("running", 3, 5), KEYS[1]: ("queued", 0, None)}

    # once every job has ended, one that failed still shows, with why
    fake_slurm.set("1001", state="OUT_OF_MEMORY", exit_code="0:125")
    fake_slurm.set("1002", state="COMPLETED")
    _progress(project, TWO[1], job="1002", state="finished", done=4, total=4)
    later = _open(project, monkeypatch)

    assert (later.run["state"], later.run["where"]) == ("finished", "the cluster")
    assert later.run["finished_at"]
    assert _states(later) == {KEYS[0]: ("failed", 3, 5), KEYS[1]: ("done", 4, 4)}
    assert later.run["items"][KEYS[0]]["problem"]["title"] == "The job ran out of its 16 GB of memory."


def test_a_rerun_while_jobs_are_queued_submits_nothing_for_them(tmp_path, raw_sessions, fake_slurm):
    from moseq2_extract import project as extraction

    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions)
    _extract(project)
    fake_slurm.set("1002", state="FAILED", exit_code="1:0")
    shutil.copytree(Path(raw_sessions) / "amphetamine_example_0", Path(project.folder) / "amphetamine_example_0")
    new = "amphetamine_example_0/depth.avi"

    planned = {p.key: p.plan for p in extraction.plan(project)}
    assert planned == {new: "new", KEYS[0]: "in progress", KEYS[1]: "new"}

    run = _extract(project)

    assert sorted(run.jobs) == sorted([new, KEYS[1]])
    assert len(fake_slurm.calls("sbatch")) == 4
    assert str(_submitted(project, TWO[0])["job_id"]) == "1001"
    assert str(_submitted(project, TWO[1])["job_id"]) in ("1003", "1004")


def test_a_failed_recording_is_resubmitted_by_a_plain_rerun(tmp_path, raw_sessions, fake_slurm):
    fake_slurm.hold()
    project = _project(tmp_path, raw_sessions, TWO[:1])
    assert _run(["extract"], project.folder).exit_code == 0
    fake_slurm.set("1001", state="OUT_OF_MEMORY", exit_code="0:125")
    _progress(project, TWO[0], job="1001", done=1, total=3, seconds_ago=5)

    result = _run(["extract"], project.folder)

    assert result.exit_code == 0, result.output
    assert str(_submitted(project, TWO[0])["job_id"]) == "1002"
    assert not (_folder(project, TWO[0]) / "extract-progress.json").exists(), "the old job's progress is cleared"
    assert "Submitted 1 job (1002)." in result.stderr

"""One run of a step at a time in a project.

``moseq2_common.project.step_lock`` holds ``<project>/.moseq2/<step>.running``
for a run's local work. A second start, from another notebook tab or a terminal,
is refused, saying where and when the first started and which process holds it,
and the first run goes on undisturbed. A lock left by a process that has ended
is taken over.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading

import pytest

from moseq2_common.errors import StepRunning
from moseq2_common.project import Project, step_lock
from extract_projects import small_project
from reporting import CollectingReporter


class _HeldReporter(CollectingReporter):
    """Holds the run at its first recording until ``release`` is set."""

    def __init__(self):
        super().__init__()
        self.reached, self.release = threading.Event(), threading.Event()

    def status(self, text):
        super().status(text)
        if text.startswith("Processing:") and not self.reached.is_set():
            self.reached.set()
            self.release.wait(60)


def test_a_second_start_while_extraction_runs_is_refused_with_who_holds_it(tmp_path, raw_sessions):
    from moseq2_extract import project as extraction

    project = small_project(tmp_path / "project", raw_sessions)
    first = _HeldReporter()
    outcome = {}

    def run():
        try:
            outcome["run"] = extraction.extract(project, reporter=first)
        except BaseException as error:  # noqa: BLE001 -- the test reports it
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    assert first.reached.wait(60), first.statuses
    try:
        with pytest.raises(StepRunning) as refused:
            extraction.extract(project, reporter=CollectingReporter())
    finally:
        first.release.set()
        thread.join(120)

    message = str(refused.value)
    assert message.startswith("Extract is already running in this project: it was started at ")
    assert f"(process {os.getpid()} on {socket.gethostname()})" in message
    assert refused.value.fix == "wait for it to finish, or stop it where it was started."
    assert "error" not in outcome, outcome
    assert outcome["run"].extracted == ["saline_example_0/depth.avi"]
    assert (tmp_path / "project" / "aggregate_results" / "saline_example_0" / "results.h5").is_file()
    assert not (tmp_path / "project" / ".moseq2" / "extract.running").exists()


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path):
    project = Project.at(tmp_path)
    ended = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    lock = tmp_path / ".moseq2" / "extract.running"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"pid": int(ended.stdout), "host": socket.gethostname(),
                                "started": "2026-11-23T09:00:00-05:00", "where": "command line"}))

    with step_lock(project, "extract", where="notebook"):
        holder = json.loads(lock.read_text())
        assert (holder["pid"], holder["where"]) == (os.getpid(), "notebook")
    assert not lock.exists()


def test_a_lock_on_another_computer_is_never_taken_over(tmp_path):
    project = Project.at(tmp_path)
    lock = tmp_path / ".moseq2" / "extract.running"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"pid": 1, "host": "another-computer", "started": "2026-11-23T09:00:00-05:00",
                                "where": "command line"}))

    with pytest.raises(StepRunning, match=r"on the command line \(process 1 on another-computer\)"):
        with step_lock(project, "extract", where="notebook"):
            pass
    assert lock.exists()

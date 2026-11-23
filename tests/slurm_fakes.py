"""A SLURM cluster for the tests: stand-ins for ``sbatch``, ``squeue``, ``sacct`` and ``scancel`` first on ``PATH``.

The ``fake_slurm`` fixture writes a launcher for each of ``tests/fake_slurm``'s
commands into a folder of the test, each running the command's script with the
test's own Python, puts that folder first on ``PATH`` and points
``FAKE_SLURM_STATE`` at a folder of the test, where the cluster keeps its jobs
(``tests/fake_slurm/_cluster.py``). No real SLURM command runs.

A submitted job starts at once as a local process running its ``--wrap``
command, unless the test holds the queue (``hold``), in which case it stays
pending until the test moves it on (``set``). ``fail`` makes a job end in a
SLURM state without running, or ``sbatch`` refuse it; ``no_accounting`` makes
``sacct`` answer as a cluster with accounting turned off. ``calls`` reads every
call the commands received. When the test ends, any job still running is
stopped.

The commands run with ``/bin/sh``, as a SLURM cluster's do, so the tests that
need them are skipped on Windows, where no SLURM cluster runs.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path

import pytest

FAKES = Path(__file__).resolve().parent / "fake_slurm"
COMMANDS = ("sbatch", "squeue", "sacct", "scancel")


class FakeSlurm:
    """The cluster a test submits to; its jobs live in ``state``."""

    def __init__(self, state: Path, bin_folder: Path, monkeypatch):
        self.state = state
        self.bin = bin_folder
        self._monkeypatch = monkeypatch

    # ---- how the cluster behaves
    def hold(self) -> None:
        """Keep every job submitted from now on in the queue, never started."""
        self._monkeypatch.setenv("FAKE_SLURM_PENDING", "1")

    def fail(self, state: str, text: str) -> None:
        """End every job whose command contains ``text`` in SLURM ``state``; ``REFUSED``: sbatch refuses it."""
        rules = [rule for rule in os.environ.get("FAKE_SLURM_FAIL", "").split(",") if rule]
        self._monkeypatch.setenv("FAKE_SLURM_FAIL", ",".join([*rules, f"{state}:{text}"]))

    def no_accounting(self) -> None:
        """Make ``sacct`` answer as a cluster with accounting turned off."""
        self._monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")

    # ---- what happened
    def jobs(self) -> dict:
        """Every job, by id."""
        folder = self.state / "jobs"
        return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))} if folder.is_dir() else {}

    def set(self, job_id: str, **values) -> None:
        """Change a job's fields, such as ``state="RUNNING"``."""
        path = self.state / "jobs" / f"{job_id}.json"
        job = json.loads(path.read_text())
        job.update(values)
        if values.get("state") not in (None, "PENDING"):
            job.setdefault("started", time.time())
        path.write_text(json.dumps(job))

    def calls(self, command: str | None = None) -> list:
        """The arguments of every call, in order; of ``command``'s only when it is given."""
        log = self.state / "calls.jsonl"
        if not log.is_file():
            return []
        calls = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        return [c["args"] if command else c for c in calls if command is None or c["command"] == command]

    def wait_for(self, check, timeout: float = 120.0) -> dict:
        """Wait until ``check(jobs)`` is true; the jobs."""
        end = time.monotonic() + timeout
        while True:
            jobs = self.jobs()
            if check(jobs):
                return jobs
            if time.monotonic() > end:
                raise AssertionError(f"the fake cluster's jobs never got there: {jobs}")
            time.sleep(0.2)

    def stop_everything(self) -> None:
        for job in self.jobs().values():
            if job.get("pid"):
                try:
                    os.killpg(job["pid"], signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """The fake cluster, its commands first on ``PATH`` (module docstring)."""
    if os.name == "nt":
        pytest.skip("SLURM clusters, and these stand-ins for their commands, run on Linux and macOS")
    state = tmp_path / "fake_slurm_state"
    bin_folder = tmp_path / "fake_slurm_bin"
    (state / "jobs").mkdir(parents=True)
    bin_folder.mkdir()
    for name in COMMANDS:
        launcher = bin_folder / name
        launcher.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKES / name))} \"$@\"\n")
        launcher.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_folder}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))
    for name in ("FAKE_SLURM_PENDING", "FAKE_SLURM_FAIL", "FAKE_SLURM_NO_SACCT"):
        monkeypatch.delenv(name, raising=False)
    cluster = FakeSlurm(state, bin_folder, monkeypatch)
    yield cluster
    cluster.stop_everything()


def submitted_lines(project) -> list[str]:
    """The ``sbatch`` line each recording's submission record holds, by session key."""
    from moseq2_common.bookkeeping import load
    from moseq2_common.yaml import read_yaml
    from moseq2_extract.recordings import session_outputs

    lines = []
    for key in sorted(load(project).sessions):
        record = session_outputs(project, key).folder / "extract-submitted.yaml"
        if record.is_file():
            lines.append(read_yaml(str(record))["sbatch"])
    return lines


def job_file_of(line: str) -> str:
    """The job settings file an ``sbatch`` line's job runs ``moseq2 run-job`` on."""
    words = shlex.split(line)
    wrapped = shlex.split(words[words.index("--wrap") + 1].rstrip().removesuffix(";"))
    assert wrapped[1:4] == ["-m", "moseq2_app", "run-job"], line
    return wrapped[4]

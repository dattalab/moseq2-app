"""What the stand-ins for ``sbatch``, ``squeue``, ``sacct`` and ``scancel`` share: a cluster kept in a folder.

The folder is ``$FAKE_SLURM_STATE``. Each job is ``jobs/<id>.json``
(``{"id", "name", "log", "command", "args", "state", "reason", "exit_code", "pid", "ends"}``),
job ids count up from 1001 (``next_id``), and every call of a command is one JSON
line in ``calls.jsonl`` (``{"command", "args"}``), so a test can count them.
Every read and write of the folder holds ``.lock``.

How the cluster behaves is set by the environment of the call:

- ``FAKE_SLURM_PENDING=1``: jobs stay in the queue (``PENDING``, reason
  ``Priority``) and never start; a test moves them on itself.
- ``FAKE_SLURM_FAIL=<state>:<text>[,<state>:<text>...]``: a job whose command
  contains ``<text>`` ends in SLURM state ``<state>`` (``OUT_OF_MEMORY``,
  ``TIMEOUT``, ``FAILED``, ``NODE_FAIL``) without running its command; with the
  state ``REFUSED``, ``sbatch`` refuses it.
- ``FAKE_SLURM_NO_SACCT=1``: ``sacct`` answers as a cluster with accounting
  turned off.

Otherwise a submitted job starts at once: a local process in a session of its
own runs the ``--wrap`` command with ``/bin/sh``, its output written to the
``-o`` file and ``SLURM_JOB_ID`` in its environment, and records ``COMPLETED``
or ``FAILED`` by its exit status when it ends.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
from pathlib import Path

#: The states a job is in while ``squeue`` lists it.
QUEUED = ("PENDING",)
ACTIVE = ("PENDING", "RUNNING")


def folder() -> Path:
    path = os.environ.get("FAKE_SLURM_STATE")
    if not path:
        sys.exit("FAKE_SLURM_STATE is not set: the fake SLURM commands keep their cluster in that folder.")
    Path(path, "jobs").mkdir(parents=True, exist_ok=True)
    return Path(path)


@contextlib.contextmanager
def locked():
    with open(folder() / ".lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def log_call(command: str, args: list[str]) -> None:
    with locked(), open(folder() / "calls.jsonl", "a") as handle:
        handle.write(json.dumps({"command": command, "args": args}) + "\n")


def job_path(job_id: str) -> Path:
    return folder() / "jobs" / f"{job_id}.json"


def load(job_id: str) -> dict | None:
    path = job_path(job_id)
    return json.loads(path.read_text()) if path.is_file() else None


def save(job: dict) -> None:
    path = job_path(job["id"])
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(job))
    os.replace(partial, path)


def next_id() -> str:
    counter = folder() / "next_id"
    value = int(counter.read_text()) if counter.is_file() else 1001
    counter.write_text(str(value + 1))
    return str(value)


def update(job_id: str, **values) -> dict | None:
    """Change a job's fields, unless it was cancelled in the meantime (a cancelled job stays cancelled)."""
    with locked():
        job = load(job_id)
        if job is None:
            return None
        if job["state"] == "CANCELLED" and values.get("state") != "CANCELLED":
            return job
        job.update(values)
        save(job)
        return job


def failures() -> list[tuple[str, str]]:
    """``FAKE_SLURM_FAIL``'s rules: (state, text) pairs."""
    rules = []
    for rule in filter(None, os.environ.get("FAKE_SLURM_FAIL", "").split(",")):
        state, _, text = rule.partition(":")
        rules.append((state.strip(), text.strip()))
    return rules


def job_ids(text: str) -> list[str]:
    return [part for part in text.split(",") if part]

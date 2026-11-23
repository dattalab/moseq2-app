"""SLURM: the options ``kappa-scan`` takes, the ``sbatch`` line, submitting jobs, and asking how they stand.

``ncpus`` is declared by the command, since its flags, default and help are
its own. Where extraction runs, on a cluster included, is its block under
``config.yaml``'s ``computing`` section (``moseq2_common.computing``).

Each job runs its step from a settings file of its own, written when the job
is submitted (``Options.to_yaml``), so a later edit to ``config.yaml`` does not
change a queued job. The file sits in the folder of what the job writes: an
extraction's in its session's folder in ``aggregate_results/``
(``extract-job.yaml``), run by ``moseq2 run-job``; a kappa-scan fit's beside its
model (``model-000-1000-job.yaml``), passed to ``learn-model --config-file``.
``job_settings_file`` names them.

A job runs MoSeq2 through the Python that submitted it (``job_command``), so it
finds the MoSeq2 the notebook or terminal runs whatever its ``PATH`` holds.
Every word of a job's command is quoted for the shell (``command_line``), and
the command is quoted again as ``--wrap``'s one argument (``sbatch_command``),
so a path with spaces or backslashes (``C:\\Users\\Jane Doe\\OneDrive``)
reaches the job as it was written.

``submit`` runs ``sbatch --parsable`` and returns the job's id; a refusal is a
``SubmitError`` carrying what ``sbatch`` said. ``job_states`` asks how jobs stand
with one ``squeue`` and, for the jobs no longer in the queue, one ``sacct``; a
job neither knows (a cluster that keeps no accounting, or no ``sacct``) is
``unknown``, and the caller judges it from what the job wrote. ``cancel`` runs
one ``scancel``. Each command runs with its output piped, never on this
process's own streams.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import click

from moseq2_common.errors import ExpectedError
from moseq2_common.options import Options, option

__all__ = ["JobState", "SlurmOptions", "SubmitError", "cancel", "command_line", "job_command", "job_settings_file",
           "job_states", "sbatch_command", "submit"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SlurmOptions(Options):
    cluster_type: str = option(
        "local", "--cluster-type", type=click.Choice(["local", "slurm"]),
        help="Platform to train models on",
    )
    prefix: str = option(
        "", "--prefix", type=str,
        help="Batch command string to prefix model training command (slurm only).",
    )
    memory: str = option("5GB", "--memory", type=str, help="RAM (slurm only)")
    wall_time: str = option("3:00:00", "--wall-time", type=str, help="Wall time (slurm only)")
    partition: str = option("short", "--partition", type=str, help="Partition name (slurm only)")
    get_cmd: bool = option(True, "--get-cmd/--no-get-cmd", is_flag=True, setting=False,
                           help="Print the commands after writing them to the script.")
    run_cmd: bool = option(False, "--run-cmd", is_flag=True, setting=False,
                           help="Run the commands written to the script.")


def job_settings_file(folder: str | os.PathLike, name: str) -> Path:
    """The settings file of the job named ``name`` (``extract``, or a model's file name without ``.p``) in ``folder``."""
    return Path(folder) / f"{name}-job.yaml"


def command_line(*words) -> str:
    """``words`` as one shell command, each quoted where the shell would otherwise split or read it."""
    return shlex.join(str(word) for word in words)


def job_command(*words: object) -> str:
    """The shell command a cluster job runs MoSeq2 with: this Python running ``moseq2_app`` with ``words``.

    ``"<sys.executable>" -m moseq2_app <words>``, each word quoted (``command_line``), so the job runs the
    MoSeq2 the notebook or terminal that submitted it runs, whatever the job's ``PATH`` holds.
    """
    return command_line(sys.executable, "-m", "moseq2_app", *words)


def sbatch_command(command: str, *, ncpus: int, memory: str, partition: str, wall_time: str, prefix: str,
                   name: str | None = None, log: str | os.PathLike | None = None, parsable: bool = False) -> str:
    """The ``sbatch`` line that runs ``command``, with ``prefix`` (shell commands of the user's) before it in the job.

    ``command`` is shell text, its paths quoted (``command_line``); the job's whole command is quoted
    as ``--wrap``'s one argument. ``parsable`` asks sbatch to answer with the job id alone, ``name`` is the
    job's name in the queue and ``log`` the file its output goes to.
    """
    return command_line(*_sbatch_words(command, ncpus=ncpus, memory=memory, partition=partition,
                                       wall_time=wall_time, prefix=prefix, name=name, log=log, parsable=parsable))


def _sbatch_words(command, *, ncpus, memory, partition, wall_time, prefix, name, log, parsable) -> list[str]:
    words = ["sbatch"]
    if parsable:
        words.append("--parsable")
    if name is not None:
        words += ["-J", name]
    if log is not None:
        words += ["-o", str(log)]
    return [*words, "-c", str(ncpus), f"--mem={memory}", "-p", partition, "-t", wall_time,
            "--wrap", f"{prefix}{command}"]


# ------------------------------------------------------------------------------- submitting and asking


class SubmitError(ExpectedError):
    """``sbatch`` refused a job, or could not be run; ``answer`` is what it said."""

    def __init__(self, message: str = "", *, answer: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.answer = answer


#: How long a SLURM command may take to answer before it counts as not answering.
TIMEOUT = 60


def _run(words: Sequence[str]) -> subprocess.CompletedProcess:
    """Run a SLURM command with its output piped; ``OSError`` (not installed) and ``TimeoutExpired`` pass up."""
    return subprocess.run(list(words), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT)


def submit(command: str, *, cpus: int, memory: str, partition: str, wall_time: str, prefix: str, name: str,
           log: str | os.PathLike) -> str:
    """Submit ``command`` as a job: ``sbatch_command(..., name=name, log=log, parsable=True)``. The job's id.

    A refusal, or an ``sbatch`` that cannot be run, raises ``SubmitError`` with what it said as ``answer``.
    """
    words = _sbatch_words(command, ncpus=cpus, memory=memory, partition=partition, wall_time=wall_time,
                          prefix=prefix, name=name, log=log, parsable=True)
    try:
        done = _run(words)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SubmitError(f"sbatch did not run: {error}", answer=str(error)) from error
    answer = (done.stderr or done.stdout or "").strip()
    job_id = done.stdout.strip().split(";")[0].strip() if done.returncode == 0 else ""
    if not job_id.isdigit():
        raise SubmitError(f"sbatch refused the job: {answer or f'exit status {done.returncode}'}",
                          answer=answer or f"nothing (exit status {done.returncode})")
    return job_id


@dataclasses.dataclass(frozen=True)
class JobState:
    """How a job stands: ``state`` is ``queued``, ``running``, ``done``, ``failed``, ``cancelled`` or ``unknown``."""

    state: str
    #: SLURM's own word for it (``PENDING``, ``OUT_OF_MEMORY``); None when SLURM no longer knows the job.
    slurm_state: str | None = None
    #: SLURM's reason, when it gives one other than ``None``.
    reason: str | None = None


#: SLURM's states, as ``job_states`` reads them; any other state ``squeue`` lists is ``running``.
_QUEUED = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD"}
_ENDED = {"COMPLETED": "done", "CANCELLED": "cancelled", "FAILED": "failed", "OUT_OF_MEMORY": "failed",
          "TIMEOUT": "failed", "NODE_FAIL": "failed", "PREEMPTED": "failed", "BOOT_FAIL": "failed",
          "DEADLINE": "failed"}


def _reason(text: str) -> str | None:
    text = text.strip()
    return None if text in ("", "None", "(null)") else text


def _answer(words: Sequence[str]) -> str | None:
    """What a SLURM command printed, or None when it could not be run or did not answer."""
    try:
        done = _run(words)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def job_states(job_ids: Sequence[str]) -> dict[str, JobState]:
    """How each of ``job_ids`` stands, by id: one ``squeue`` for the jobs queued or running, then one ``sacct``
    for the others. A job neither command knows, because the cluster keeps no accounting, ``sacct`` is not
    installed or it did not answer, is ``unknown``."""
    ids = [str(job_id) for job_id in job_ids]
    if not ids:
        return {}
    states: dict[str, JobState] = {}
    # squeue fails outright when it knows none of the ids, as for jobs that have all ended
    for line in (_answer(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T %r"]) or "").splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0] in ids:
            word = parts[1].upper()
            states[parts[0]] = JobState("queued" if word in _QUEUED else "running", word,
                                        _reason(parts[2] if len(parts) > 2 else ""))
    gone = [job_id for job_id in ids if job_id not in states]
    if gone:
        accounting = _answer(["sacct", "-n", "-P", "-j", ",".join(gone), "-o", "JobID,State,ExitCode,Reason"])
        for line in (accounting or "").splitlines():
            parts = line.split("|")
            if len(parts) < 2 or parts[0] not in gone:
                continue  # a job's steps (1001.batch) and anything else asked for
            word = parts[1].split()[0].upper() if parts[1].strip() else ""
            if word in _ENDED:
                state = _ENDED[word]
            elif word in _QUEUED:
                state = "queued"
            elif word:
                state = "running"
            else:
                continue
            states[parts[0]] = JobState(state, word, _reason(parts[3] if len(parts) > 3 else ""))
    return {job_id: states.get(job_id, JobState("unknown")) for job_id in ids}


def cancel(job_ids: Sequence[str]) -> None:
    """Cancel ``job_ids`` with one ``scancel``. A job that has ended already is left as it is; an ``scancel``
    that cannot be run raises ``SubmitError``."""
    ids = [str(job_id) for job_id in job_ids]
    if not ids:
        return
    try:
        _run(["scancel", *ids])
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SubmitError(f"scancel did not run: {error}", answer=str(error)) from error

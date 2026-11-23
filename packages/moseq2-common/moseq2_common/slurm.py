"""The SLURM options ``kappa-scan`` takes, the ``sbatch`` line, and cluster jobs' commands and settings files.

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
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import sys
from pathlib import Path

import click

from moseq2_common.options import Options, option

__all__ = ["SlurmOptions", "command_line", "job_command", "job_settings_file", "sbatch_command"]


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


def sbatch_command(command: str, *, ncpus: int, memory: str, partition: str,
                   wall_time: str, prefix: str) -> str:
    """The ``sbatch`` line that runs ``command``, with ``prefix`` (shell commands of the user's) before it in the job.

    ``command`` is shell text, its paths quoted (``command_line``); the job's whole command is quoted
    as ``--wrap``'s one argument.
    """
    return command_line("sbatch", "-c", ncpus, f"--mem={memory}", "-p", partition, "-t", wall_time,
                        "--wrap", f"{prefix}{command}")

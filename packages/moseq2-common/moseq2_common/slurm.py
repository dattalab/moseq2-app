"""The SLURM options ``batch-extract`` and ``kappa-scan`` share, and their sbatch line.

``ncpus`` is declared by each command, since its flags, default and help
differ between the two.
"""

from __future__ import annotations

import dataclasses

import click

from moseq2_common.options import Options, option

__all__ = ["SlurmOptions", "sbatch_command"]


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


def sbatch_command(command: str, *, ncpus: int, memory: str, partition: str,
                   wall_time: str, prefix: str) -> str:
    """The ``sbatch`` line that runs ``command``, with ``prefix`` before it in the job."""
    return f'sbatch -c {ncpus} --mem={memory} -p {partition} -t {wall_time} --wrap "{prefix}{command}"'

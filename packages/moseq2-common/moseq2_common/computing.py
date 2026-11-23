"""Where a step runs: this computer or a SLURM cluster, and what it may use there.

Each step that does long work has a block of its own under ``config.yaml``'s
top-level ``computing`` section, keyed by the step's command name::

    computing:
      extract:
        cluster_type: local
        cores: null
        partition: short
        memory: 16GB
        wall_time: auto
        prefix: ''

so two steps that share their other settings can still run in different places.
``ComputingOptions`` declares the fields; each step subclasses it with its
``STEP`` and its own defaults, and its package lists the subclass in its options
module's ``COMPUTING`` tuple, from which ``generate-config`` writes the blocks.
``moseq2_common.options.read_settings`` reads a computing class from
``computing.<STEP>`` and ``save_settings`` writes it there.

``cores`` None means every core this process may use (``available_cores``).
``memory`` and ``wall_time`` keep SLURM's spellings (``16GB``, ``1:30:00``);
``gigabytes`` and ``hours`` read them as numbers, and ``wall_time`` ``auto``
leaves the time to the step.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import ClassVar

import click

from moseq2_common.options import Control, Options, option

__all__ = ["ComputingOptions", "available_cores", "gigabytes", "hours"]


def available_cores() -> int:
    """How many CPU cores this process may use: its affinity where the system has one, else every core."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


_MEMORY = re.compile(r"\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)B?\s*", re.IGNORECASE)
#: Gigabytes per unit; a bare number is megabytes, as SLURM reads it.
_PER_GIGABYTE = {"": 1 / 1024, "K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def gigabytes(memory: str) -> float:
    """``memory`` in gigabytes: ``"16GB"`` is 16.0, ``"512MB"`` 0.5, a bare number megabytes."""
    match = _MEMORY.fullmatch(str(memory))
    if match is None:
        raise ValueError(f"{memory!r} is not an amount of memory. Write it as SLURM does, such as 16GB or 512MB.")
    return float(match[1]) * _PER_GIGABYTE[match[2].upper()]


def hours(wall_time: str) -> float | None:
    """``wall_time`` in hours: ``"1:30:00"`` is 1.5; ``"auto"`` is None (the step decides).

    SLURM's forms are read: ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and ``D-HH:MM:SS``.
    """
    text = str(wall_time).strip()
    if text == "auto":
        return None
    days = 0
    if "-" in text:
        day, text = text.split("-", 1)
        days = int(day)
        parts = [int(p) for p in text.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts
    else:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = parts
    return days * 24 + h + m / 60 + s / 3600


#: The two places a step can run, as the GUI offers them.
_PLACES = (("local", "This computer"), ("slurm", "SLURM cluster"))


@dataclass(frozen=True, kw_only=True)
class ComputingOptions(Options):
    """Where a step runs and what it may use there; each step subclasses it with its ``STEP``."""

    SECTION = "computing"
    #: The command the block belongs to (``extract``), its key under ``computing``.
    STEP: ClassVar[str] = ""

    cluster_type: str = option(
        "local", "--cluster-type", type=click.Choice(["local", "slurm"]),
        help="Where the work runs: on this computer (local), or as jobs on a SLURM cluster (slurm).",
        control=Control("segmented", "Run on", choices=_PLACES,
                        help="Where the work runs: on this computer, or as jobs on a SLURM cluster."))
    cores: int | None = option(
        None, "--cores", type=int,
        help="How many of this computer's CPU cores it may use; by default, every core it may use.",
        control=Control("number", "CPU cores", minimum=1, slider=True, show_if=("cluster_type", "local")))
    partition: str = option(
        "short", "--partition", type=str,
        help="The cluster queue the jobs go to, as your cluster names it.",
        control=Control("text", "Partition", placeholder="partition name", show_if=("cluster_type", "slurm")))
    memory: str = option(
        "16GB", "--memory", type=str,
        help="How much memory each cluster job asks for, such as 16GB.",
        control=Control("number", "Memory per job", minimum=2, maximum=512, unit="GB", shown_as="gigabytes",
                        help="How much memory each cluster job asks for.",
                        show_if=("cluster_type", "slurm")))
    wall_time: str = option(
        "auto", "--wall-time", type=str,
        help="How long each cluster job may run before the cluster stops it, as hours:minutes:seconds "
             "(such as 4:00:00), or auto to set it from the work each job has.",
        control=Control("choice", "Time per job", shown_as="hours", show_if=("cluster_type", "slurm"),
                        help="How long each cluster job may run before the cluster stops it."))
    prefix: str = option(
        "", "--prefix", type=str,
        help="Shell commands each cluster job runs before its work, such as loading a module or activating "
             "an environment.")

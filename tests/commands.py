"""Split a command line moseq2 writes back into its arguments.

A SLURM extraction records each recording's ``sbatch`` line, and ``kappa-scan``
writes one per fit into a script, with paths as arguments. Tests run each
written command to check what the job would do, so they split it first.
"""

from __future__ import annotations

import os
import shlex


def split_command(command, windows=os.name == "nt"):
    """The arguments of ``command``, with each path as it was written.

    A POSIX split reads a backslash as an escape and drops it, which turns a
    Windows path such as ``C:\\Users\\me\\depth.avi`` into
    ``C:Usersmedepth.avi``. On Windows the split keeps backslashes, and only
    takes the quotes off a quoted argument.
    """
    if not windows:
        return shlex.split(command)
    return [_unquote(arg) for arg in shlex.split(command, posix=False)]


def _unquote(arg):
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
        return arg[1:-1]
    return arg

"""Split a command moseq2 writes into a script back into its arguments.

The SLURM path of ``batch-extract`` and of the notebook's extraction cell
writes one ``moseq2-extract extract`` command per recording into
``extract_out.sh``, with the config file and the recording as paths. Tests run
each written command through ``extract`` to check what the job would extract
with, so they split it first.
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

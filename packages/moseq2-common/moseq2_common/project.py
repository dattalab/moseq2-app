"""A MoSeq2 project: the folder holding ``config.yaml``.

A command finds its project from ``--config-file``'s folder, or from the folder
it runs in when that holds a ``config.yaml`` (``find_project``). With neither it
runs standalone: its outputs go where its arguments say, and one line says so
(``standalone_note``).

A job settings file, which ``batch-extract`` and ``kappa-scan`` write for each
job they submit, is a settings file in ``config.yaml``'s sectioned shape with a
``job:`` block naming the project folder (``write_job_settings``). It sits in a
session's or the models' folder, so a step run from it takes its project from the
block, never from where the file is.

Per-recording settings live in ``session_config.yaml`` (``SESSION_CONFIG_NAME``)
in the project folder, found without a pointer in ``config.yaml``. Its entries
are keyed by session: the recording's path relative to the project folder, with
``/`` between folders on every system (``session_key``), so a project moved or
copied elsewhere still finds them.

A ``config.yaml`` without sections was written by MoSeq2 1.5.0 or earlier.
1.6.0 does not read or convert it, and stops with ``OldProjectError``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath

from moseq2_common.yaml import read_yaml, write_config

__all__ = ["CONFIG_NAME", "SESSION_CONFIG_NAME", "OldProjectError", "Project", "find_project", "is_job_file",
           "old_project_message", "session_key", "standalone_note", "write_job_settings"]

CONFIG_NAME = "config.yaml"
#: The file holding each recording's arena-tool settings, in the project folder.
SESSION_CONFIG_NAME = "session_config.yaml"


class OldProjectError(Exception):
    """A project made by MoSeq2 1.5.0 or earlier, which 1.6.0 does not read."""


def old_project_message(file: Path, reason: str) -> str:
    """What to do with the project ``file`` shows is from 1.5.0 or earlier, ``reason`` saying how it shows it."""
    file = Path(file)
    return (
        f"{file} shows that {file.parent} is a project made with MoSeq2 1.5.0 or earlier ({reason}), "
        "and MoSeq2 1.6.0 does not read or convert those projects. To finish this project, install the "
        "version it was started with in a separate environment: "
        '`python -m pip install "moseq2-app==1.5.0"`. To use 1.6.0, make a new project folder holding '
        "(or linking to) the raw recordings and start there with the 1.6 notebooks or "
        "`moseq2-extract generate-config`."
    )


@dataclass(frozen=True)
class Project:
    """A project folder and the files in it that hold its state."""

    folder: Path
    config_file: Path
    bookkeeping_file: Path
    session_config_file: Path

    @classmethod
    def at(cls, folder: str | os.PathLike) -> Project:
        """The project in ``folder``: its files' paths only, nothing read."""
        folder = Path(folder)
        return cls(folder=folder, config_file=folder / CONFIG_NAME,
                   bookkeeping_file=folder / "moseq2-project.yaml",
                   session_config_file=folder / SESSION_CONFIG_NAME)


def session_key(project: Project, recording: str | os.PathLike) -> str:
    """The session a recording is: its path relative to ``project``'s folder, ``/`` between folders.

    ``cohort1/saline_example_4/depth.avi`` on every system. A recording outside the
    project folder is keyed by its path from there (``../elsewhere/depth.avi``), or by
    its absolute path where no relative path exists (another drive on Windows).
    """
    folder = os.path.abspath(project.folder)
    path = os.path.abspath(recording)
    try:
        relative = os.path.relpath(path, folder)
    except ValueError:
        relative = path
    return "/".join(PurePath(relative).parts)


def _job_project(config_file: str | os.PathLike):
    """The project folder a job settings file's ``job:`` block names; None for a file without the block.

    A file that cannot be read has no block here; reading its settings names the problem.
    """
    try:
        data = read_yaml(config_file)
    except Exception:  # noqa: BLE001 -- read_settings reports an unreadable file, with what to do
        return None
    if not isinstance(data, Mapping) or not isinstance(data.get("job"), Mapping):
        return None
    return data["job"]


def is_job_file(config_file: str | os.PathLike | None) -> bool:
    """Whether ``config_file`` is a job settings file (it has a ``job:`` block).

    A job's settings already hold its recording's ``session_config.yaml`` entry,
    so a step run from one does not lay the entry over them again.
    """
    return bool(config_file) and _job_project(config_file) is not None


def find_project(config_file: str | os.PathLike | None = None,
                 cwd: str | os.PathLike | None = None) -> Project | None:
    """The project a command runs in.

    With ``config_file``: the project its ``job:`` block names (None when the
    block names none), else the file's folder. Without: ``cwd`` (the working
    folder by default) when it holds a ``config.yaml``, else None.
    """
    if config_file:
        job = _job_project(config_file)
        if job is not None:
            return Project.at(job["project"]) if job.get("project") else None
        return Project.at(Path(config_file).parent)
    folder = Path(cwd) if cwd is not None else Path.cwd()
    if (folder / CONFIG_NAME).is_file():
        return Project.at(folder)
    return None


def standalone_note(cwd: str | os.PathLike) -> str:
    """The line a command prints when it finds no project."""
    return (f"No config.yaml in {cwd} and none given with --config-file, so this runs outside a project: "
            "outputs go where the arguments say and no moseq2-project.yaml is updated.")


def write_job_settings(path: str | os.PathLike, section: str, settings: Mapping, *,
                       project: Project | None) -> Path:
    """Write one job's settings to ``path``, in config.yaml's shape, with the ``job:`` block.

    ``settings`` go in ``section``; a ``camera_type`` among them goes to the top
    level, where config.yaml holds it. The block names ``project``'s folder, or
    holds ``project: null`` for a job submitted outside a project.
    """
    settings = dict(settings)
    data = {}
    if "camera_type" in settings:
        data["camera_type"] = settings.pop("camera_type")
    data[section] = settings
    data["job"] = {"project": str(Path(project.folder).absolute()) if project is not None else None}
    write_config(data, str(path))
    return Path(path)

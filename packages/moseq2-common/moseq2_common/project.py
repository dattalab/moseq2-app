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

Groups are a setting: ``config.yaml``'s top-level ``groups`` maps each session
key to its group, and its ``exclude`` lists the sessions every step after
extraction leaves out. ``write_groups`` is their one writer, used by
``add-group`` and the notebook's group widget. A key in ``groups``, ``exclude``
or ``session_config.yaml`` that names no session of the project is a warning
(``check_session_names``), never an error.

What the project holds and where each step's outputs are is in its bookkeeping
file, ``moseq2-project.yaml`` (``moseq2_common.bookkeeping``).

A project made by MoSeq2 1.5.0 or earlier shows itself by a ``config.yaml``
without sections, or by 1.5.0's ``progress.yaml`` or ``moseq2-index.yaml`` with
no ``moseq2-project.yaml`` beside it. 1.6.0 does not read or convert one:
``check_not_old`` stops on it with ``OldProjectError``, and every command, setup
command and the notebooks' project cell call it first. A name a 1.5 notebook
imports that 1.6.0 removed stops the import with ``OutdatedNotebookError``
(``outdated_names``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath

from moseq2_common.yaml import read_yaml, write_config

__all__ = ["CONFIG_NAME", "SESSION_CONFIG_NAME", "OldProjectError", "OutdatedNotebookError", "Project",
           "check_not_old", "find_project", "is_job_file", "old_project_message", "outdated_names",
           "outdated_notebook_message", "check_session_names", "exclude_note", "read_exclude", "read_groups",
           "session_key", "standalone_note", "write_groups",
           "write_job_settings"]

CONFIG_NAME = "config.yaml"
#: The file holding each recording's arena-tool settings, in the project folder.
SESSION_CONFIG_NAME = "session_config.yaml"


class OldProjectError(Exception):
    """A project made by MoSeq2 1.5.0 or earlier, which 1.6.0 does not read."""


class OutdatedNotebookError(ImportError):
    """A 1.5 notebook importing a name MoSeq2 1.6.0 removed."""


#: 1.5.0's files that show a project was made by it, when no ``moseq2-project.yaml`` is beside them,
#: and the words the message gives for each.
_OLD_FILES = (("progress.yaml", "it is 1.5.0's progress file"),
              ("moseq2-index.yaml", "it is 1.5.0's index file"))


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


def _has_no_sections(config_file: Path) -> bool:
    """Whether ``config_file`` holds settings outside any section, as 1.5.0's did.

    A file that cannot be read is not judged here; reading its settings names the problem.
    """
    from moseq2_common.options import SECTIONS, TOP_LEVEL_KEYS

    try:
        data = read_yaml(config_file)
    except Exception:  # noqa: BLE001 -- read_settings reports an unreadable file, with what to do
        return False
    return (isinstance(data, Mapping) and not any(name in data for name in SECTIONS)
            and any(key not in TOP_LEVEL_KEYS for key in data))


def check_not_old(folder: str | os.PathLike) -> None:
    """Stop with ``OldProjectError`` when ``folder`` holds a project made by MoSeq2 1.5.0 or earlier.

    That is a ``config.yaml`` without sections, or a ``progress.yaml`` or
    ``moseq2-index.yaml`` with no ``moseq2-project.yaml`` beside it. The message
    names the file that showed it. Nothing in the folder is written.
    """
    folder = Path(folder)
    config = folder / CONFIG_NAME
    if config.is_file() and _has_no_sections(config):
        raise OldProjectError(old_project_message(config, "its settings have no sections"))
    if (folder / "moseq2-project.yaml").is_file():
        return
    for name, reason in _OLD_FILES:
        if (folder / name).is_file():
            raise OldProjectError(old_project_message(folder / name, reason))


def outdated_notebook_message(module: str, name: str) -> str:
    """What to do when a 1.5 notebook imports ``module.name``, which 1.6.0 removed."""
    return (
        f"This notebook was written for MoSeq2 1.5 and uses `{module}.{name}`, which MoSeq2 1.6.0 does not "
        "have. Download the 1.6 notebooks from https://github.com/dattalab/moseq2-app/tree/release/notebooks "
        "and start from their first cell. A project the 1.5 notebooks started is finished on 1.5.0: "
        '`python -m pip install "moseq2-app==1.5.0"`.'
    )


def outdated_names(module: str, names: Iterable[str]) -> Callable[[str], object]:
    """A module ``__getattr__`` under which each of ``names`` stops with ``OutdatedNotebookError``.

    ``names`` are what the 1.5 notebooks import from ``module`` that 1.6.0
    removed; any other missing name is the usual ``AttributeError``.
    """
    names = frozenset(names)

    def __getattr__(name: str):
        if name in names:
            raise OutdatedNotebookError(outdated_notebook_message(module, name))
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__


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


def read_groups(project: Project) -> dict[str, str]:
    """``config.yaml``'s ``groups``: each session key's group. Empty without a config.yaml or groups."""
    from moseq2_common.options import read_config

    if not Path(project.config_file).is_file():
        return {}
    groups = read_config(project.config_file).get("groups") or {}
    if not isinstance(groups, Mapping):
        raise ValueError(
            f"{project.config_file}'s `groups` holds {type(groups).__name__}, not one `session: group` line per "
            "session. Correct it, or delete it and set the groups again with the notebook's group widget or "
            "`moseq2-viz add-group`."
        )
    return {str(key): str(value) for key, value in groups.items()}


def read_exclude(project: Project) -> list[str]:
    """``config.yaml``'s ``exclude``: the session keys every step after extraction leaves out.

    Empty without a config.yaml or an ``exclude``.
    """
    from moseq2_common.options import read_config

    if not Path(project.config_file).is_file():
        return []
    exclude = read_config(project.config_file).get("exclude") or []
    if isinstance(exclude, str):
        exclude = [exclude]
    if not isinstance(exclude, list):
        raise ValueError(
            f"{project.config_file}'s `exclude` holds {type(exclude).__name__}, not a list of sessions. Write "
            "it as one `- session` line per session to leave out (each session's recording path in the "
            "project, as `groups` names it), or delete it to leave no session out."
        )
    return [str(key) for key in exclude]


def exclude_note(keys: Iterable[str]) -> str:
    """The line a step prints when it leaves out the sessions ``keys``, which ``exclude`` lists."""
    keys = list(keys)
    return f"Leaving out the {len(keys)} sessions under exclude in config.yaml: {', '.join(keys)}."


def check_session_names(project: Project, sessions: Iterable[str] | None = None) -> list[str]:
    """A warning for each key in ``groups``, ``exclude`` or ``session_config.yaml`` that is not a session.

    ``sessions`` are the keys of the project's sessions; None reads them from its
    bookkeeping file. Before any session is listed there is nothing to compare
    with, and no warning. A ``session_config.yaml`` that cannot be read is not
    judged here; extraction reads it and names the problem.
    """
    if sessions is None:
        from moseq2_common.bookkeeping import load

        sessions = load(project).sessions
    sessions = set(sessions)
    if not sessions:
        return []  # nothing aggregated yet to compare the names with
    named = [(project.config_file, key) for key in [*read_groups(project), *read_exclude(project)]]
    if Path(project.session_config_file).is_file():
        try:
            entries = read_yaml(str(project.session_config_file))
        except Exception:  # noqa: BLE001 -- extraction reports an unreadable session_config.yaml
            entries = None
        if isinstance(entries, Mapping):
            named += [(project.session_config_file, str(key)) for key in entries]
    return [f"{file} names {key}, which is not a recording in this project. It is ignored; correct or delete "
            "the entry if the recording moved."
            for file, key in named if key not in sessions]


def write_groups(project: Project, groups: Mapping[str, str], exclude: Iterable[str] | None = None) -> None:
    """Set ``config.yaml``'s ``groups`` to ``groups``, and its ``exclude`` to ``exclude`` when given.

    The rest of the file stays as it was.
    """
    from moseq2_common.options import read_config

    data = read_config(project.config_file)
    data["groups"] = {str(key): str(value) for key, value in groups.items()}
    if exclude is not None:
        data["exclude"] = [str(key) for key in exclude]
    write_config(data, str(project.config_file))

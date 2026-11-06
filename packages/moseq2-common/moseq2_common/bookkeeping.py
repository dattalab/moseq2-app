"""A project's bookkeeping file, ``moseq2-project.yaml``: what the project holds and where.

It sits in the project folder beside ``config.yaml`` and lists:

- the sessions, each keyed by its recording's path relative to the project
  (``moseq2_common.project.session_key``), with the recording, its folder in
  ``aggregate_results/`` (``folder``, holding the extraction ``results.h5``,
  ``SESSION_FILE``) and its uuid;
- each step's current outputs: the PCA components, scores and changepoints, the
  fitted models and which one is current, each model's analysis folder, and the
  applied models.

It holds nothing that is a setting (groups are in ``config.yaml``) and nothing
copied from an output. Every path in it is relative to the project folder, with
``/`` between folders on every system, so a project moved or copied elsewhere
keeps working. Only the process running a step writes it, at the end of the
step, through a temporary file renamed into place (``update``); a step run from
a SLURM job's settings file never does.

The viz and model steps take the sessions as an index in memory
(``session_index``), the shape ``moseq2_viz.util.parse_index`` reads, built from
this file and ``config.yaml``'s groups each time and never written. It leaves
out the sessions ``config.yaml``'s ``exclude`` lists, so every step after
extraction does, and prints one line naming them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from os.path import abspath, dirname
from pathlib import Path, PurePosixPath

from moseq2_common.project import Project, check_session_names, exclude_note, read_exclude, read_groups, session_key
from moseq2_common.yaml import read_yaml, write_yaml

__all__ = ["BOOKKEEPING_NAME", "SESSION_FILE", "Bookkeeping", "BookkeepingError", "Session", "load",
           "project_index", "relative_path", "resolve_path", "session_h5", "session_index", "sessions_folder",
           "update"]

BOOKKEEPING_NAME = "moseq2-project.yaml"
#: A session's extraction file, in its folder; its preview movie and results yaml share the name.
SESSION_FILE = "results.h5"

#: The group of a session ``config.yaml``'s ``groups`` does not name.
DEFAULT_GROUP = "default"


class BookkeepingError(ValueError):
    """A bookkeeping file that cannot be read, or that lacks what a step needs."""


@dataclass
class Session:
    #: The recording's path relative to the project folder; the session's name everywhere.
    key: str
    #: The recording, relative to the project folder.
    recording: str
    #: The session's folder in ``aggregate_results/``, relative to the project folder.
    folder: str
    uuid: str


@dataclass
class Bookkeeping:
    sessions: dict[str, Session] = field(default_factory=dict)
    #: ``components``, ``scores``, ``changepoints``: each a path relative to the project folder.
    pca: dict[str, str] = field(default_factory=dict)
    #: ``fitted`` (a list of models), ``current`` (one of them, or None), ``analysis`` ({model: folder}).
    models: dict[str, object] = field(default_factory=lambda: {"fitted": [], "current": None, "analysis": {}})
    applied_models: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sessions": {key: {"recording": s.recording, "folder": s.folder, "uuid": s.uuid}
                         for key, s in self.sessions.items()},
            "pca": dict(self.pca),
            "models": {"fitted": list(self.models.get("fitted") or []), "current": self.models.get("current"),
                       "analysis": dict(self.models.get("analysis") or {})},
            "applied_models": list(self.applied_models),
        }


def relative_path(project: Project, path: str | os.PathLike) -> str:
    """``path`` as the bookkeeping file stores it: relative to the project folder, ``/`` between folders."""
    return session_key(project, path)


def resolve_path(project: Project, stored: str) -> Path:
    """A path the bookkeeping file stores, as a path on this system."""
    stored_path = PurePosixPath(stored)
    if stored_path.is_absolute() or os.path.isabs(stored):
        return Path(stored)
    return Path(abspath(Path(project.folder).joinpath(*stored_path.parts)))


def _unreadable(project: Project, problem: str) -> BookkeepingError:
    return BookkeepingError(
        f"{project.bookkeeping_file} {problem}, so MoSeq2 cannot tell what this project holds. It is "
        f"written by the pipeline's steps; if it was edited by hand, undo the edit, or delete it and run "
        f"`moseq2-extract aggregate-results` in {project.folder} (or the notebook's aggregation cell) to "
        "list the sessions again, then run the steps after it again."
    )


def load(project: Project) -> Bookkeeping:
    """The project's bookkeeping file; an empty one when the file does not exist yet."""
    path = Path(project.bookkeeping_file)
    if not path.exists():
        return Bookkeeping()
    try:
        data = read_yaml(str(path))
    except Exception as error:  # noqa: BLE001 -- any parse failure is the same problem for the user
        raise _unreadable(project, f"could not be read ({error})") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise _unreadable(project, "does not hold the sessions and outputs as `name: value` lines")
    sessions = {}
    for key, entry in (data.get("sessions") or {}).items():
        entry = entry or {}
        sessions[key] = Session(key=key, recording=entry.get("recording", key), folder=entry.get("folder"),
                                uuid=entry.get("uuid"))
    models = {"fitted": [], "current": None, "analysis": {}}
    models.update(data.get("models") or {})
    return Bookkeeping(sessions=sessions, pca=dict(data.get("pca") or {}), models=models,
                       applied_models=list(data.get("applied_models") or []))


def update(project: Project, change: Callable[[Bookkeeping], None]) -> Bookkeeping:
    """Read the bookkeeping file, apply ``change`` to it, and write it through a temporary file."""
    bookkeeping = load(project)
    change(bookkeeping)
    path = Path(project.bookkeeping_file)
    temporary = path.with_name(f".{path.name}.tmp")
    write_yaml(bookkeeping.as_dict(), str(temporary))
    os.replace(temporary, path)
    return bookkeeping


def session_h5(project: Project, session: Session) -> Path:
    """The extraction file of ``session``: ``results.h5`` in its folder."""
    return resolve_path(project, session.folder) / SESSION_FILE


def sessions_folder(project: Project, bookkeeping: Bookkeeping | None = None) -> Path:
    """The folder holding the project's session files (``aggregate_results/``)."""
    bookkeeping = bookkeeping if bookkeeping is not None else load(project)
    if not bookkeeping.sessions:
        raise BookkeepingError(
            f"{project.bookkeeping_file if Path(project.bookkeeping_file).exists() else project.folder} lists no "
            "sessions yet. Aggregate the extractions first: run `moseq2-extract aggregate-results` in "
            f"{project.folder}, or the notebook's aggregation cell."
        )
    folders = {dirname(resolve_path(project, s.folder)) for s in bookkeeping.sessions.values()}
    return Path(os.path.commonpath(sorted(folders)))


def _metadata(h5: Path, fallback: str) -> dict:
    """The acquisition metadata the session's results yaml holds, beside its ``.h5``."""
    yaml_file = h5.with_suffix(".yaml")
    try:
        stored = (read_yaml(str(yaml_file)) or {}).get("metadata")
    except OSError:
        stored = None
    metadata = dict(stored) if isinstance(stored, dict) else {}
    metadata.setdefault("SessionName", fallback)
    metadata.setdefault("SubjectName", fallback)
    return metadata


def session_index(project: Project, *, exclude: bool = True) -> dict:
    """The project's sessions as an index in memory: 1.5.0's parsed-index shape, absolute paths.

    ``{"files": [{"uuid", "group", "metadata", "path": (h5, yaml), "key"}], "pca_path": scores or "",
    "bookkeeping_file": the file it was read from, "excluded": {key: uuid}}`` (``excluded`` only when
    sessions are left out),
    each session's group from ``config.yaml``'s ``groups`` (``default`` when it names none). Sessions are
    listed in the order they were recorded (``StartTime``), then by name.

    With ``exclude`` (the default), the sessions ``config.yaml``'s ``exclude`` lists are left out of
    ``files``, listed in ``excluded`` and named in one printed line; the group widget and ``add-group``
    pass False to list every session. A key in ``groups``, ``exclude`` or ``session_config.yaml`` that
    is not a session of the project is printed as a warning (``check_session_names``).
    """
    bookkeeping = load(project)
    groups = read_groups(project)
    for warning in check_session_names(project, bookkeeping.sessions):
        print(warning)
    left_out = [key for key in read_exclude(project) if key in bookkeeping.sessions] if exclude else []
    if left_out:
        print(exclude_note(left_out))
    entries = []
    for key, session in bookkeeping.sessions.items():
        if key in left_out:
            continue
        h5 = session_h5(project, session)
        entries.append({
            "uuid": session.uuid,
            "group": groups.get(key, DEFAULT_GROUP),
            "metadata": _metadata(h5, key),
            "path": (str(h5), str(h5.with_suffix(".yaml"))),
            "key": key,
        })
    # recording order: the start time to the second, as 1.5.0's aggregated file names began with it
    entries.sort(key=lambda f: (str(f["metadata"].get("StartTime") or "")[:19],
                                str(f["metadata"].get("SessionName")), f["key"]))
    scores = bookkeeping.pca.get("scores")
    index = {"files": entries, "pca_path": str(resolve_path(project, scores)) if scores else "",
             "bookkeeping_file": str(project.bookkeeping_file)}
    if left_out:
        index["excluded"] = {key: bookkeeping.sessions[key].uuid for key in left_out}
    return index


def project_index(project: Project | None) -> dict | None:
    """``session_index`` of ``project``; None outside a project or before any session is aggregated."""
    if project is None or not load(project).sessions:
        return None
    return session_index(project)

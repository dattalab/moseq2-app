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

The PCA, model and viz steps take the sessions as an index in memory
(``session_index``), the shape ``moseq2_viz.util.parse_index`` reads, built from
this file and ``config.yaml``'s groups each time and never written. It leaves
out the sessions ``config.yaml``'s ``exclude`` lists, so every step after
extraction does, and prints one line naming them. ``exclude`` is a setting like
any other: a list given for one run (``--exclude``, ``--no-exclude``, a step's
``exclude=``) replaces ``config.yaml``'s for that run, and ``groups=`` likewise
replaces its groups. ``extraction_keys`` names the extractions found in a folder
by their session keys, so the list applies to a typed ``--input-dir`` too.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from os.path import abspath, dirname
from pathlib import Path, PurePosixPath

from moseq2_common.h5 import extraction_labels
from moseq2_common.project import Project, check_session_names, exclude_note, read_exclude, read_groups, session_key
from moseq2_common.records import InputError, StepInputError
from moseq2_common.yaml import read_yaml, write_yaml

__all__ = ["BOOKKEEPING_NAME", "SESSION_ADVICE", "SESSION_FILE", "Bookkeeping", "BookkeepingError", "Session",
           "check_given_exclude", "exclude_line", "extraction_keys", "index_for_run", "load", "note_sessions",
           "project_index", "recording_order", "relative_path", "resolve_path", "session_h5", "session_index",
           "sessions_folder", "update"]

BOOKKEEPING_NAME = "moseq2-project.yaml"
#: A session's extraction file, in its folder; its preview movie shares the name.
SESSION_FILE = "results.h5"

#: What to do when a session's extraction, which a step after extraction reads, is unusable.
SESSION_ADVICE = ("Extract that session again (moseq2-extract extract, or the notebook's extraction cell), then "
                  "run this step again; or, to leave the session out, list its recording under exclude in "
                  "config.yaml (the group widget's exclude column).")

#: The group of a session ``config.yaml``'s ``groups`` does not name.
DEFAULT_GROUP = "default"


class BookkeepingError(StepInputError, ValueError):
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
    """The acquisition metadata the session's extraction holds (``metadata/acquisition`` in its ``.h5``)."""
    try:
        _, metadata = extraction_labels(h5)
    except OSError:
        metadata = {}
    metadata.setdefault("SessionName", fallback)
    metadata.setdefault("SubjectName", fallback)
    return metadata


def recording_order(metadata: dict | None, name: str) -> tuple:
    """The sort key that lists sessions in the order they were recorded: the acquisition ``StartTime`` to the
    second (as 1.5.0's aggregated file names began with it), then ``SessionName``, then ``name`` (the
    session's key, or its file when it has none). Every step lists sessions in this order; PCA's changepoints
    depend on it, since they draw one random projection per session in turn."""
    metadata = metadata or {}
    return str(metadata.get("StartTime") or "")[:19], str(metadata.get("SessionName") or name), name


def exclude_line(left_out, *, given: bool, in_project: bool = True) -> str | None:
    """The line a run prints naming the sessions it leaves out, ``left_out``: from ``config.yaml``'s ``exclude``,
    or, when ``given``, the list given for this run. None when there is nothing to say (none left out, none
    given)."""
    left_out = list(left_out)
    if not given:
        return exclude_note(left_out) if left_out else None
    instead = ", in place of config.yaml's exclude" if in_project else ""
    if not left_out:
        return f"Leaving out no session for this run{instead}."
    count = "the 1 session" if len(left_out) == 1 else f"the {len(left_out)} sessions"
    return f"Leaving out {count} given for this run{instead}: {', '.join(left_out)}."


def check_given_exclude(given, keys, where) -> None:
    """Stop when a session key in the list ``given`` for this run is not one of ``keys``, the sessions it
    chooses among (named in ``where``): a typo would otherwise leave the session in."""
    keys = list(keys)
    unknown = [key for key in given if key not in keys]
    if not unknown:
        return
    shown = ", ".join(keys[:5]) + (f" and {len(keys) - 5} more" if len(keys) > 5 else "")
    raise InputError(
        f"The sessions to leave out given for this run name {', '.join(unknown)}, which "
        f"{'is not a session' if len(unknown) == 1 else 'are not sessions'} here. A session is named by its "
        f"recording's path in its project, as {where} lists it ({shown}). Correct the name, or leave it out."
    )


def session_index(project: Project, *, exclude=None, groups=None) -> dict:
    """The project's sessions as an index in memory: 1.5.0's parsed-index shape, absolute paths.

    ``{"files": [{"uuid", "group", "metadata", "path": (h5,), "key"}], "pca_path": scores or "",
    "bookkeeping_file": the file it was read from, "excluded": {key: uuid}, "exclude": [key]}``
    (``excluded`` only when sessions are left out, ``exclude`` only when a list was given),
    each session's group from ``config.yaml``'s ``groups``, or from ``groups`` when given (``default`` for a
    session it names none for). Sessions are listed in the order they were recorded (``StartTime``), then by
    name.

    ``exclude`` None leaves out the sessions ``config.yaml``'s ``exclude`` lists; a list given (``[]`` for
    none) replaces it, and a key in it that is not a session stops (``check_given_exclude``). The sessions
    left out are listed in ``excluded`` and named in one printed line (``exclude_line``), printed also for a
    list given empty. ``exclude=False`` lists every session and prints no line: the group widget and
    ``add-group`` list them so. A key in ``groups``, ``exclude`` or ``session_config.yaml``
    that is not a session of the project is printed as a warning (``check_session_names``).
    """
    bookkeeping = load(project)
    groups = read_groups(project) if groups is None else {str(key): str(value) for key, value in groups.items()}
    for warning in check_session_names(project, bookkeeping.sessions):
        print(warning)
    if exclude is False:
        left_out = []
    elif exclude is None:
        left_out = [key for key in read_exclude(project) if key in bookkeeping.sessions]
    else:
        left_out = list(dict.fromkeys(str(key) for key in exclude))
        check_given_exclude(left_out, bookkeeping.sessions, project.bookkeeping_file)
    line = exclude_line(left_out, given=exclude is not None) if exclude is not False else None
    if line:
        print(line)
    entries = []
    for key, session in bookkeeping.sessions.items():
        if key in left_out:
            continue
        h5 = session_h5(project, session)
        entries.append({
            "uuid": session.uuid,
            "group": groups.get(key, DEFAULT_GROUP),
            "metadata": _metadata(h5, key),
            # a sequence whose first item is the extraction, the shape the viz and model readers take (1.5.0's
            # (h5, yaml) pair; extraction writes no yaml)
            "path": (str(h5),),
            "key": key,
        })
    entries.sort(key=lambda f: recording_order(f["metadata"], f["key"]))
    scores = bookkeeping.pca.get("scores")
    index = {"files": entries, "pca_path": str(resolve_path(project, scores)) if scores else "",
             "bookkeeping_file": str(project.bookkeeping_file)}
    if left_out:
        index["excluded"] = {key: bookkeeping.sessions[key].uuid for key in left_out}
    if exclude is not None and exclude is not False:
        index["exclude"] = left_out
    return index


def project_index(project: Project | None, *, exclude=None, groups=None) -> dict | None:
    """``session_index`` of ``project``; None outside a project or before any session is aggregated.

    A list to leave out given outside a project stops, since there are no sessions for it to name.
    """
    if project is None:
        if exclude:
            raise InputError(
                f"The sessions to leave out given for this run ({', '.join(exclude)}) name sessions of a project, "
                "and this runs outside one. Run it from the project folder (or name the project with "
                "--project-file), or leave --exclude out."
            )
        return None
    if not load(project).sessions:
        return None
    return session_index(project, exclude=exclude, groups=groups)


def index_for_run(index, *, exclude=None, groups=None):
    """``index`` with ``exclude`` and ``groups`` in place of ``config.yaml``'s, for a step given an index and one
    of them; ``index`` itself when neither is given, or when it was already built with that ``exclude``.

    The index is built again from its project (``session_index``), so it must be a project's.
    """
    if exclude is None and groups is None:
        return index
    if groups is None and isinstance(index, dict) and index.get("exclude") == list(dict.fromkeys(exclude)):
        return index
    if not isinstance(index, dict) or not index.get("bookkeeping_file"):
        raise InputError(
            "exclude= and groups= choose among a project's sessions, and the sessions given are not a project's. "
            "Pass the project's sessions, session_index(Project.at(project_folder)), or leave exclude and groups "
            "out."
        )
    return session_index(Project.at(Path(index["bookkeeping_file"]).parent), exclude=exclude, groups=groups)


def note_sessions(record, index) -> None:
    """Note in ``record`` the sessions of ``index`` a run used and those it left out, by key."""
    if not isinstance(index, dict) or not isinstance(index.get("files"), list) or not index.get("bookkeeping_file"):
        return
    record.note("sessions", [entry.get("key") for entry in index["files"]])
    record.note("exclude", list(index.get("excluded") or {}))


def _owning_project(path: Path) -> Project | None:
    """The project whose folder holds ``path`` and a ``moseq2-project.yaml``, nearest first."""
    for folder in path.parents:
        if (folder / BOOKKEEPING_NAME).is_file():
            return Project.at(folder)
    return None


def extraction_keys(h5s, uuids, project: Project | None, folder) -> list[str]:
    """The session key of each extraction in ``h5s`` (with its ``uuids``), found under ``folder``.

    An extraction that is a session of ``project`` (the running project), by its path or its uuid, is that
    session; one that is a session of the project holding it (another project, ``--input-dir`` pointed into
    it) is that one's session, keyed as that project keys it. Any other is keyed by its path relative to
    ``folder``, ``/`` between folders.
    """
    tables = {}

    def table(owner):
        key = str(Path(owner.folder).resolve())
        if key not in tables:
            sessions = load(owner).sessions
            tables[key] = ({str(session_h5(owner, s).resolve()): k for k, s in sessions.items()},
                           {s.uuid: k for k, s in sessions.items()})
        return tables[key]

    keys = []
    for h5, uuid in zip(h5s, uuids):
        path = Path(h5).resolve()
        found = None
        if project is not None:
            by_path, by_uuid = table(project)
            found = by_path.get(str(path)) or by_uuid.get(uuid)
        owner = _owning_project(path) if found is None else None
        if owner is not None:
            found = table(owner)[0].get(str(path))
        keys.append(found or Path(os.path.relpath(path, Path(folder).resolve())).as_posix())
    return keys

"""
Find the depth recordings in a project.

The progress cell, the arena tool and extraction (the notebook cell and
``moseq2-extract batch-extract``) all list a project's recordings through
``find_recordings``, so they agree on what a recording is.

A recording is one file per session folder whose name ends in one of the
extensions the caller asks for (``.dat``, ``.avi``, ``.mkv``, ``.tar.gz``), at
most ``max_depth`` folders below the project folder, following symlinked
folders. IR videos (``ir.dat``, ``ir.avi``), the NI-DAQ stream (``nidaq.dat``)
and flipped movies (``*_flipped.*``) sit beside recordings and are not
recordings. A session folder holding more than one recording keeps one: the
one with "depth" in its name, then the first in ``.dat``, ``.mkv``, ``.avi``,
``.tar.gz`` order. Each ``.tar.gz`` (or ``.tgz``) archive is a session of its
own and carries its ``metadata.json`` inside it.

A session is its recording, keyed by the recording's path relative to the
project folder (``moseq2_common.project.session_key``). Inside a project, every
file the pipeline writes for a session goes into the session's own folder in
``aggregate_results/`` (``session_outputs``), named from its key
(``session_folder_name``): nothing is written beside or into a recording. Outside
a project, ``moseq2-extract extract FILE`` writes into ``proc/`` beside the
recording. A recording is extracted when its ``results.h5`` exists under that
name; extraction writes it under a temporary name and renames it when done.
"""

import hashlib
import os
from collections.abc import Collection
from dataclasses import dataclass, field
from os.path import abspath, exists, isdir, join, realpath
from pathlib import Path, PurePosixPath

from moseq2_common.bookkeeping import SESSION_FILE, load, resolve_path
from moseq2_common.project import Project, session_key

#: The recording types the pipeline reads, in the order one is preferred over
#: another when a session folder holds several.
RECORDING_EXTENSIONS = (".dat", ".mkv", ".avi", ".tar.gz")

#: How many folders below the project folder the search goes by default.
DEFAULT_MAX_DEPTH = 5

#: The folder in a project holding one folder per session.
SESSIONS_FOLDER = "aggregate_results"
#: The folder beside a recording that ``extract FILE`` writes into outside a project.
STANDALONE_FOLDER = "proc"

_ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz")


def _normalize_extension(ext):
    ext = ext.strip()
    return ext if ext.startswith(".") else "." + ext


def _matching_extension(name, extensions):
    """The extension in ``extensions`` that ``name`` ends with, or None."""
    for ext in sorted(extensions, key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return None


def _stem(name):
    for ext in sorted(RECORDING_EXTENSIONS + _ARCHIVE_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def is_not_a_recording(name):
    """True for the files that sit beside a recording with a recording's extension:
    IR videos, the NI-DAQ stream and flipped movies."""
    stem = _stem(name).lower()
    is_ir = stem == "ir" or stem.startswith(("ir_", "ir-")) or stem.endswith(("_ir", "-ir"))
    return is_ir or stem == "nidaq" or stem.endswith("_flipped")


def _preference(name):
    """Sort key choosing the recording a session folder keeps."""
    ext = _matching_extension(name, RECORDING_EXTENSIONS + (".tgz",)) or ""
    order = RECORDING_EXTENSIONS.index(ext) if ext in RECORDING_EXTENSIONS else len(RECORDING_EXTENSIONS)
    return ("depth" not in name.lower(), order, name)


def _join_words(items, last="or"):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


@dataclass
class RecordingSearch:
    """What ``find_recordings`` found under ``root``, and what it passed over.

    Attributes:
    root (str): the folder searched.
    extensions (tuple): the extensions searched for, with their leading dots.
    max_depth (int): how many folders below ``root`` the search went.
    recordings (list): paths of the recordings found, sorted.
    without_metadata (list): session folders holding a found recording but no metadata.json.
    passed_over (list): (folder, reason) for each folder left out, in the order met.
    extracted (list): (recording, its extraction file) for each recording left out as already extracted.
    too_deep (list): folders whose subfolders lie deeper than ``max_depth`` and were not searched.
    """

    root: str
    extensions: tuple
    max_depth: int
    recordings: list = field(default_factory=list)
    without_metadata: list = field(default_factory=list)
    passed_over: list = field(default_factory=list)
    extracted: list = field(default_factory=list)
    too_deep: list = field(default_factory=list)
    root_exists: bool = True

    def warnings(self):
        """One line per session folder whose recording has no metadata.json beside it."""
        return [
            f"Warning: {folder} has no metadata.json. Its recording {name} is included, and its "
            "extraction gets an empty subject name and the folder's name as its session name; "
            "nothing is written into the folder. If the acquisition software wrote a metadata.json "
            f"for this recording, copy it into {folder} before extracting so the subject and "
            "session names are kept."
            for folder, name in self.without_metadata
        ]

    def report(self):
        """What a user needs to know about this search, or an empty string when nothing was left out.

        Says where the search looked, how deep, what counts as a recording and
        which folders it passed over and why, whenever it found nothing or passed
        a folder over; names how many recordings were left out as already
        extracted whenever any were; and warns for each recording without a
        metadata.json. When every recording found is already extracted, one line
        says so, and the rest follows only when a folder was passed over.
        """
        lines = self.warnings()
        if not self.root_exists:
            lines += [
                f"No recordings found: {self.root} is not a folder. Check the path given as the "
                "project folder (base_dir in the notebook, the input folder on the command line)."
            ]
            return "\n".join(lines)

        extracted_line = (
            f"{len(self.extracted)} recording(s) already extracted were left out, each with its "
            f"extraction file written (such as {self.extracted[0][1]}). To include "
            "them, set skip_extracted to False (in the notebook) or add --skip-checks "
            "(moseq2-extract batch-extract)."
        ) if self.extracted else None
        left_out = self.passed_over or self.too_deep
        if self.recordings and not left_out:
            return "\n".join(lines + ([extracted_line] if extracted_line else []))

        all_extracted = not self.recordings and bool(self.extracted)
        if all_extracted:
            lines.append(
                f"All {len(self.extracted)} recordings under {self.root} are already extracted, so there is "
                "nothing to extract. To extract them again, set skip_extracted to False (in the notebook) "
                "or add --skip-checks (moseq2-extract batch-extract)."
            )
            if not left_out:
                return "\n".join(lines)
            lines.append(
                f"Searched {self.root} and its folders down to {self.max_depth} levels below it, "
                "following symlinked folders."
            )
        elif self.recordings:
            lines.append(
                f"Found {len(self.recordings)} recording(s) under {self.root}, searching its folders "
                f"down to {self.max_depth} levels below it and following symlinked folders."
            )
        else:
            lines.append(f"No recordings found under {self.root}.")
            lines.append(
                f"Searched {self.root} and its folders down to {self.max_depth} levels below it, "
                "following symlinked folders."
            )
        lines.append(
            f"A recording is a file ending in {_join_words(self.extensions)}, one per session folder. "
            "IR videos (ir.dat, ir.avi), the NI-DAQ stream (nidaq.dat) and flipped movies "
            "(*_flipped.*) are not recordings."
        )

        passed_over = list(self.passed_over)
        passed_over += [
            (
                folder,
                f"its subfolders are more than {self.max_depth} levels below {self.root} and were not searched",
            )
            for folder in self.too_deep
        ]
        if passed_over:
            lines.append("Folders passed over:")
            lines += [f"  {folder}: {reason}" for folder, reason in passed_over]
        if extracted_line and not all_extracted:
            lines.append(extracted_line)
        if self.too_deep:
            deeper = self.max_depth + 5
            lines.append(
                f"To search deeper, pass a larger max_depth, for example max_depth={deeper} "
                f"(--max-depth {deeper} on the moseq2-extract batch-extract command line)."
            )
        if not self.recordings and not self.extracted:
            lines.append(
                f"Put each session's recording in its own folder under {self.root}, or point the "
                "search at the folder that holds them."
            )
        return "\n".join(lines)


def find_recordings(
    root,
    extensions=RECORDING_EXTENSIONS,
    max_depth=DEFAULT_MAX_DEPTH,
    extracted=None,
):
    """
    Find the depth recordings under ``root``.

    Args:
    root (str): the project folder.
    extensions (iterable): recording extensions to look for, with or without the leading dot.
    max_depth (int): how many folders below ``root`` to search; ``root`` itself is level 0.
    extracted (callable): to leave out the recordings already extracted, a function taking the list of
        every recording found and returning {recording: its extraction file} for those extracted
        (``extracted_in``); None leaves none out.

    Returns:
    (RecordingSearch): the recordings found, sorted, and what was passed over and why;
        ``report()`` gives the message for the user.
    """
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = tuple(dict.fromkeys(_normalize_extension(e) for e in extensions))
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError(
            f"max_depth must be a whole number of folders, 0 or more; got {max_depth!r}. "
            f"The default, {DEFAULT_MAX_DEPTH}, searches five folders below the project folder."
        )

    root = abspath(root)
    search = RecordingSearch(root=root, extensions=extensions, max_depth=max_depth)
    if not isdir(root):
        search.root_exists = False
        return search

    # a .tgz is the same archive as a .tar.gz
    searched = extensions + ((".tgz",) if ".tar.gz" in extensions else ())
    seen_folders = set()
    seen_recordings = set()
    found = []

    for folder, dirs, files in os.walk(root, followlinks=True):
        real = realpath(folder)
        level = 0 if folder == root else os.path.relpath(folder, root).count(os.sep) + 1
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if real in seen_folders:
            # reached again through a symlink; its recordings are already listed
            dirs[:] = []
            continue
        seen_folders.add(real)
        if level >= max_depth:
            if [d for d in dirs if d != STANDALONE_FOLDER]:
                search.too_deep.append(folder)
            dirs[:] = []

        typed = [f for f in sorted(files) if _matching_extension(f, RECORDING_EXTENSIONS + (".tgz",))]
        if not typed:
            continue
        wanted = [f for f in typed if _matching_extension(f, searched)]
        other_types = [f for f in typed if f not in wanted and not is_not_a_recording(f)]
        candidates = [f for f in wanted if not is_not_a_recording(f)]
        archives = [f for f in candidates if f.endswith(_ARCHIVE_EXTENSIONS)]
        plain = sorted((f for f in candidates if f not in archives), key=_preference)

        if not candidates:
            if other_types:
                search.passed_over.append(
                    (
                        folder,
                        f"holds {', '.join(other_types)}, but this search looks only for {', '.join(extensions)}; "
                        "add the extension to the ones searched to include it",
                    )
                )
            elif wanted:
                search.passed_over.append(
                    (folder, f"holds only {', '.join(wanted)}, which are not depth recordings")
                )
            continue

        kept = archives + plain[:1]
        if len(plain) > 1:
            search.passed_over.append(
                (
                    folder,
                    f"holds more than one recording; took {plain[0]} and left out {', '.join(plain[1:])} "
                    "(a session folder holds one recording: one with \"depth\" in its name is taken first, "
                    "then .dat, .mkv, .avi in that order). If a file left out is a different recording, "
                    "move it into a folder of its own",
                )
            )

        for name in kept:
            path = join(folder, name)
            if realpath(path) in seen_recordings:
                continue
            seen_recordings.add(realpath(path))
            if name not in archives and not exists(join(folder, "metadata.json")):
                if (folder, name) not in search.without_metadata:
                    search.without_metadata.append((folder, name))
            found.append(path)

    done = extracted(sorted(found)) if extracted is not None else {}
    search.extracted = [(path, str(done[path])) for path in sorted(found) if path in done]
    search.without_metadata = [(folder, name) for folder, name in search.without_metadata
                               if join(folder, name) not in done]
    search.recordings = sorted(path for path in found if path not in done)
    return search


def session_folder_name(key: str, keys: Collection[str]) -> str:
    """The name of session ``key``'s folder in ``aggregate_results/``, among the sessions ``keys``.

    The recording's folder path with ``/`` as ``_`` (``cohort1/saline_example_4/depth.avi`` is
    ``cohort1_saline_example_4``); the recording's own name, without its extension, is added when its
    folder holds more than one recording of ``keys`` (several archives), and is the name of a recording
    in the project folder itself. A name that two keys still share gets the first six characters of
    each key's SHA-1 appended, so two sessions never share a folder and a name does not depend on the
    order of ``keys``.
    """
    def parts(k):
        return [part for part in PurePosixPath(k.replace("\\", "/")).parts if part not in ("", ".", "..", "/")]

    def flat(k):
        *folder, name = parts(k) or [k]
        siblings = [other for other in keys if parts(other)[:-1] == folder] if folder else [k]
        stem = _stem(name)
        pieces = folder + ([stem] if len(set(siblings) | {k}) > 1 or not folder else [])
        return "_".join(piece.replace(":", "_") for piece in pieces)

    keys = set(keys) | {key}
    name = flat(key)
    if any(flat(other) == name for other in keys if other != key):
        name = f"{name}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:6]}"
    return name


@dataclass(frozen=True)
class SessionOutputs:
    """Where one session's outputs are: its folder and the files extraction writes in it."""

    folder: Path
    #: The extraction (``results.h5``); a session is extracted when this exists.
    h5: Path
    #: The extraction's preview movie (``results.mp4``).
    movie: Path
    #: The images extraction writes: ``background``, ``first_frame``, and with their settings
    #: ``plane_background`` and ``graduated_background``.
    images: dict


def outputs_in(folder) -> SessionOutputs:
    """The names extraction writes in ``folder``."""
    folder = Path(folder)
    h5 = folder / SESSION_FILE
    return SessionOutputs(folder=folder, h5=h5, movie=h5.with_suffix(".mp4"), images={
        "background": folder / "bground.tiff", "first_frame": folder / "first_frame.tiff",
        "plane_background": folder / "plane_bground.tiff", "graduated_background": folder / "new_bg.tiff"})


def project_keys(project: Project, max_depth: int = DEFAULT_MAX_DEPTH) -> list:
    """The session keys of every recording in ``project``'s folder."""
    return [session_key(project, r) for r in find_recordings(project.folder, max_depth=max_depth).recordings]


def session_outputs(project: Project, key: str, *, keys: Collection[str] | None = None,
                    sessions_dir=None) -> SessionOutputs:
    """Where session ``key``'s outputs are in ``project``.

    The folder ``moseq2-project.yaml`` names for the session when it lists it; otherwise
    ``session_folder_name`` of the key among ``keys`` (by default, every recording in the
    project), in ``sessions_dir`` (by default the project's ``aggregate_results/``). A
    ``sessions_dir`` given is used as given.
    """
    if sessions_dir is None:
        listed = load(project).sessions.get(key)
        if listed is not None and listed.folder:
            return outputs_in(resolve_path(project, listed.folder))
        sessions_dir = Path(project.folder) / SESSIONS_FOLDER
    if keys is None:
        keys = project_keys(project)
    return outputs_in(Path(sessions_dir) / session_folder_name(key, keys))


def extracted_in(project: Project, sessions_dir=None):
    """The ``extracted`` function ``find_recordings`` takes: each recording whose ``results.h5`` exists."""
    def extracted(recordings):
        keys = {session_key(project, r): r for r in recordings}
        everything = set(keys) | set(project_keys(project))
        done = {}
        for key, recording in keys.items():
            h5 = session_outputs(project, key, keys=everything, sessions_dir=sessions_dir).h5
            if h5.is_file():
                done[recording] = h5
        return done

    return extracted


def standalone_outputs(recording) -> SessionOutputs:
    """Where ``extract FILE`` writes outside a project: ``proc/`` beside the recording."""
    return outputs_in(Path(abspath(recording)).parent / STANDALONE_FOLDER)

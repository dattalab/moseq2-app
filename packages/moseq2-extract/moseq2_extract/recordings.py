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

A session is named by its folder's name, or by its folder's path below the
project folder when two session folders share a name (``session_keys``).
``session_config.yaml`` is keyed by those names, and ``session_config_key``
finds a recording's entry from the recording's path alone.
"""

import os
from dataclasses import dataclass, field
from os.path import abspath, basename, exists, isdir, join, realpath
from pathlib import PurePath

#: The recording types the pipeline reads, in the order one is preferred over
#: another when a session folder holds several.
RECORDING_EXTENSIONS = (".dat", ".mkv", ".avi", ".tar.gz")

#: How many folders below the project folder the search goes by default.
DEFAULT_MAX_DEPTH = 5

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


def _completed(status_file):
    from moseq2_extract.helpers.data import check_completion_status

    return check_completion_status(status_file)


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
    extracted (list): (recording, results yaml) for each recording left out as already extracted.
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
            f"Warning: {folder} has no metadata.json. Its recording {name} is included, and a "
            "default metadata.json, with the session named after the folder, is written there "
            "when the recording is extracted or listed by the progress cell. If the acquisition "
            f"software wrote a metadata.json for this recording, copy it into {folder} first so "
            "the subject and session names are kept."
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
            f"{len(self.extracted)} recording(s) already extracted were left out, each with a "
            f"finished extraction recorded beside it (such as {self.extracted[0][1]}). To include "
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
    skip_extracted=False,
    results_yaml=join("proc", "results_00.yaml"),
):
    """
    Find the depth recordings under ``root``.

    Args:
    root (str): the project folder.
    extensions (iterable): recording extensions to look for, with or without the leading dot.
    max_depth (int): how many folders below ``root`` to search; ``root`` itself is level 0.
    skip_extracted (bool): leave out recordings whose ``results_yaml`` records a finished extraction.
    results_yaml (str): where an extraction's results yaml sits, relative to the recording's folder.

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
    results_folder = results_yaml.split(os.sep)[0] if os.sep in results_yaml else None

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
            if [d for d in dirs if d != results_folder]:
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
            if skip_extracted and _completed(join(folder, results_yaml)):
                search.extracted.append((path, join(folder, results_yaml)))
                continue
            if name not in archives and not exists(join(folder, "metadata.json")):
                if (folder, name) not in search.without_metadata:
                    search.without_metadata.append((folder, name))
            found.append(path)

    search.recordings = sorted(found)
    return search


def session_keys(recordings, root):
    """
    Name each recording's session folder, as ``session_config.yaml`` and the notebook's lists name it.

    A folder is named by its own name when no other recording's folder has that
    name, which is every folder in most projects. Folders that share a name
    (``GroupA/mouse1`` and ``GroupB/mouse1``) are each named by their path below
    ``root``, with ``/`` between folders on every system, so that each keeps its
    own parameters.

    Args:
    recordings (iterable): paths of recordings, from ``find_recordings``.
    root (str): the project folder they were found under.

    Returns:
    (dict): the name of each recording's session folder, by recording path.
    """
    root = abspath(root)
    folders = {r: abspath(os.path.dirname(abspath(r))) for r in recordings}
    owners = {}
    for folder in set(folders.values()):
        owners.setdefault(basename(folder), []).append(folder)

    def key(folder):
        if len(owners[basename(folder)]) == 1 or folder == root:
            return basename(folder)
        return "/".join(PurePath(os.path.relpath(folder, root)).parts)

    return {r: key(f) for r, f in folders.items()}


def _key_parts(key):
    return tuple(part for part in str(key).replace("\\", "/").split("/") if part)


def session_config_key(session_configs, recording, by_folder_name=True):
    """
    The entry of ``session_config.yaml`` that holds a recording's parameters.

    An entry named by a path (``GroupA/mouse1``, written when two session
    folders share a name) is the recording's when the recording's folder path
    ends with it; the longest such entry is taken. Otherwise, with
    ``by_folder_name``, the entry named by the folder's own name is. The project
    folder is not needed, so extraction, which is handed a recording and the
    config, finds the entry the arena tool wrote.

    Args:
    session_configs (dict): ``session_config.yaml`` as read.
    recording (str): path to the recording.
    by_folder_name (bool): fall back to the entry named by the folder's own name.

    Returns:
    (str or None): the entry's key, or None when there is none.
    """
    if not session_configs:
        return None
    folder = PurePath(abspath(os.path.dirname(abspath(recording)))).parts
    best = None
    for key in session_configs:
        parts = _key_parts(key)
        if len(parts) < 2 or len(parts) > len(folder):
            continue
        if folder[-len(parts):] == parts and (best is None or len(parts) > len(_key_parts(best))):
            best = key
    if best is not None:
        return best
    name = basename(os.path.dirname(abspath(recording)))
    if by_folder_name and name in session_configs:
        return name
    return None

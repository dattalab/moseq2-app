"""Label files, the step that saves them, and what each one labels.

Syllable labels (``syll_info.yaml``) are keyed by usage-sorted syllable number, so
they describe one fit of one model; the Flip-Classifier-Training notebook's saved
frame ranges (``flip-training-frame-ranges.p``) describe one state of each
session's frames. A label file holds the labels only.

Saving labels is a step with its own record (``save_record``), written beside the
labels file (``moseq2_common.records.sidecar_path``: ``syll_info.record.yaml``,
``flip-training-frame-ranges.yaml``). It carries the save time and, as its inputs,
what the labels label with the run that wrote it (``moseq2_common.records.run_of``):
the model for syllable labels, each labelled session's extraction for the frame
ranges.

Labels are stale when the file they label now carries the record of another run:
a model fit again, a session extracted again or flipped by a flip classifier, or
another model altogether (``stale``). A tool opening stale labels does not attach
them: it says so (``stale_labels_message``) and offers two choices, to wipe the
labels and continue, or to stop so the person can copy the labels somewhere
first. The pipeline keeps no old label file. Labels with no record beside them
have nothing to check against and are opened as they are. Nothing here reads a
file's modification time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from moseq2_common.project import Project
from moseq2_common.records import read_record, run_of, sidecar_path, start_record, unchanged, write_record, written_on
from moseq2_common.yaml import read_yaml

__all__ = ["FLIP_LABELS_STEP", "SYLLABLE_LABELS_STEP", "WIPE", "STOP", "index_project", "labelled_in",
           "read_syllable_labels", "remove_labels", "save_record", "stale", "stale_labels_message",
           "stale_syllable_labels", "stopped_message"]

#: The steps that save labels, as their records name them.
SYLLABLE_LABELS_STEP = "save-syllable-labels"
FLIP_LABELS_STEP = "save-flip-labels"

#: The two choices offered for stale labels, as their buttons read.
WIPE = "Wipe labels and continue"
STOP = "Stop"


def index_project(index) -> Project | None:
    """The project a session index (``moseq2_common.bookkeeping.session_index``) was read from, or None."""
    if isinstance(index, Mapping) and index.get("bookkeeping_file"):
        return Project.at(Path(index["bookkeeping_file"]).parent)
    return None


def save_record(labels_file: str | os.PathLike, step: str, labelled: Sequence, project: Project | None = None) -> Path:
    """Write the record of saving ``labels_file`` beside it: ``labelled``, the files the labels label (paths,
    or mappings holding ``path`` and facts such as ``session``), as its inputs. Returns the record's file."""
    record = start_record(step, {}, labelled, project=project)
    return write_record(labels_file, record.finish({"files": [str(labels_file)]}))


def labelled_in(labels_file: str | os.PathLike) -> list[dict]:
    """The files the record beside ``labels_file`` names as labelled; empty when it has no record."""
    try:
        record = read_record(labels_file)
    except FileNotFoundError:
        return []
    return [entry for entry in record.get("inputs") or [] if isinstance(entry, Mapping)]


def remove_labels(labels_file: str | os.PathLike) -> None:
    """Delete ``labels_file`` and its record."""
    record = sidecar_path(labels_file)  # before the file goes: a missing file's sidecar is named otherwise
    Path(labels_file).unlink(missing_ok=True)
    record.unlink(missing_ok=True)


def read_syllable_labels(path: str | os.PathLike) -> dict:
    """The syllable labels in ``path`` (``syll_info.yaml``), by syllable number."""
    return dict(read_yaml(str(path)) or {})


def _relative(path, project: Project) -> str:
    """``path`` relative to ``project``'s folder, as a record's ``relative_path`` holds it."""
    return Path(os.path.relpath(os.path.abspath(path), Path(project.folder).resolve())).as_posix()


def stale(entry: Mapping, path: str | os.PathLike, verb: str, different: str | None = None) -> str | None:
    """Why labels saved for ``entry`` (an input of their record) do not fit ``path`` now, as a clause of
    ``stale_labels_message``; None when they do.

    ``verb`` says how the labelled file was written (``fit``, ``extracted``). ``different`` is the reason to
    give instead when ``path`` is not the file ``entry`` names (``is a different model``).
    """
    if different is None and unchanged(entry, path):
        return None
    then = entry.get("run") or {}
    labelled = entry.get("path", "another file")
    made = f"{labelled} as {verb} on {written_on(then['ended'])}" if then.get("ended") else str(labelled)
    if different is not None:
        return f"{made}, and {path} {different}"
    now = run_of(path)
    if now is None or not now.get("ended"):
        reason = "has changed since, and carries no record of the run that wrote it"
    elif (now.get("flip_corrected") and now.get("flip_corrected") != then.get("flip_corrected")
          and (now.get("started"), now.get("ended")) == (then.get("started"), then.get("ended"))):
        reason = f"was flipped by a flip classifier on {written_on(now['flip_corrected'])}"
    else:
        reason = f"was {verb} again on {written_on(now['ended'])}"
    return f"{made}, and {path} {reason}"


def stale_syllable_labels(labels_file: str | os.PathLike, model_path: str | os.PathLike,
                          project: Project | None = None) -> str | None:
    """Why the syllable labels in ``labels_file`` do not fit ``model_path`` now (``stale``), or None."""
    entries = labelled_in(labels_file)
    if not entries:
        return None
    entry = entries[0]
    if project is not None and entry.get("relative_path"):
        same = entry["relative_path"] == _relative(model_path, project)
    else:
        same = entry.get("path") == os.path.abspath(model_path)
    return stale(entry, os.path.abspath(model_path), "fit", None if same else "is a different model")


def stale_labels_message(labels: str | os.PathLike, found: list[str]) -> str:
    """The message for labels in ``labels`` that do not fit what they label (``stale`` results)."""
    return (f"The labels in {labels} were made for {'; for '.join(found)}. Wipe the labels and continue, or stop "
            f"and copy {labels} somewhere safe first.")


def stopped_message(labels: str | os.PathLike) -> str:
    """What a tool says when the person chooses to stop."""
    return (f"Stopped. {labels} is unchanged. Copy it somewhere safe, then run this cell again and choose "
            f"{WIPE}.")

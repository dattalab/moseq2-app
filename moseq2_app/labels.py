"""Label files, and the file each one labels.

Syllable labels (``syll_info.yaml``) are keyed by usage-sorted syllable number, so
they describe one fit of one model; the Flip-Classifier-Training notebook's saved
frame ranges (``flip-training-frame-ranges.p``) describe one state of each
session's frames. Each label file records what it labels under ``labels_for``:
the labelled file's path and its fingerprint (``moseq2_common.records.fingerprint``),
one entry for ``syll_info.yaml``'s model and one per labelled session for the
frame ranges.

A tool opening labels whose labelled file no longer has that fingerprint, or is
newer than the labels, does not attach them: it says so (``stale_labels_message``)
and offers two choices, to wipe the labels and continue, or to stop so the person
can copy the labels somewhere first. The pipeline keeps no old label file. Labels
that record nothing under ``labels_for`` have nothing to check against and are
opened as they are.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime

from moseq2_common.records import fingerprint
from moseq2_common.yaml import read_yaml

__all__ = ["LABELS_FOR", "WIPE", "STOP", "labels_for", "read_syllable_labels", "stale", "stale_labels_message",
           "stopped_message"]

#: The key a label file records what it labels under.
LABELS_FOR = "labels_for"

#: The two choices offered for stale labels, as their buttons read.
WIPE = "Wipe labels and continue"
STOP = "Stop"


def labels_for(path: str | os.PathLike) -> dict:
    """What a label file records about the file it labels: its path and its fingerprint."""
    return {"path": os.path.abspath(path), "fingerprint": fingerprint(path)}


def read_syllable_labels(path: str | os.PathLike) -> dict:
    """The syllable labels in ``path`` (``syll_info.yaml``), by syllable number, without ``labels_for``."""
    labels = dict(read_yaml(str(path)) or {})
    labels.pop(LABELS_FOR, None)
    return labels


def _when(stored: Mapping) -> str:
    modified = (stored.get("fingerprint") or {}).get("modified")
    return str(modified)[:19].replace("T", " ") if modified else "an unrecorded time"


def stale(stored: Mapping | None, path: str | os.PathLike, different: str) -> tuple[str, str, str, str] | None:
    """``(labelled, when, current, reason)`` when labels made for ``stored`` (a ``labels_for`` entry) do not
    fit ``path`` now, else None. ``different`` is the reason given when ``path`` is not newer than the labels
    but is not what they were made for (``is a different model``)."""
    if not stored or stored.get("fingerprint") == fingerprint(path):
        return None
    recorded = (stored.get("fingerprint") or {}).get("modified")
    try:
        newer = recorded is None or os.stat(path).st_mtime > datetime.fromisoformat(recorded).timestamp()
    except (TypeError, ValueError):
        newer = False
    reason = "is newer than the labels" if newer else different
    return str(stored.get("path", "another file")), _when(stored), os.path.abspath(path), reason


def stale_labels_message(labels: str | os.PathLike, found: list[tuple[str, str, str, str]]) -> str:
    """The message for labels in ``labels`` that do not fit what they label (``stale`` results)."""
    made_for = "; for ".join(f"{labelled} ({when}), and {current} {reason}" for labelled, when, current, reason
                             in found)
    return (f"The labels in {labels} were made for {made_for}. Wipe the labels and continue, or stop and copy "
            f"{labels} somewhere safe first.")


def stopped_message(labels: str | os.PathLike) -> str:
    """What a tool says when the person chooses to stop."""
    return (f"Stopped. {labels} is unchanged. Copy it somewhere safe, then run this cell again and choose "
            f"{WIPE}.")

"""The step GUIs open on each project, so a change one makes shows in every other.

Each GUI registers itself when it opens (``opened``). When a GUI saves settings
or a run ends, ``changed(project)`` asks every GUI open on that project to read
it again (``StepWidget.refresh``). GUIs are held weakly: a closed one is
forgotten.
"""

from __future__ import annotations

import os
import weakref
from pathlib import Path

_OPEN: dict[str, weakref.WeakSet] = {}


def _key(project) -> str:
    return os.path.realpath(Path(project.folder))


def opened(widget) -> None:
    """Remember ``widget`` as open on its project."""
    _OPEN.setdefault(_key(widget.project), weakref.WeakSet()).add(widget)


def changed(project) -> None:
    """Refresh every GUI open on ``project``; one that cannot is left as it was."""
    for widget in list(_OPEN.get(_key(project), ())):
        try:
            widget.refresh()
        except Exception as error:  # noqa: BLE001 -- shown in that GUI
            widget.report_problem(error)

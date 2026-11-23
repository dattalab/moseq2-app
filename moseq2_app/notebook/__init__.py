"""The pipeline notebook's steps: each cell is ``from moseq2_app.notebook import <step>`` then ``<step>()``.

Each function opens its step's GUI on the project in the kernel's working folder,
where the notebook sits: ``setup``, ``arena_tool``, ``extract``, ``diagnostics``,
``train_pca`` and ``scores_and_changepoints``. Running a cell only opens the GUI;
its buttons do the work, in the background, through the step's project layer
(``moseq2_<package>.project``), the same functions ``moseq2 <step>`` calls.

Each step lives in a module of its own here (``_STEPS``), imported when its
function is first asked for. ``_widget`` holds what every step's GUI shares,
``_state`` the GUIs open on each project, and ``static/`` their scripts and styles.
"""

from __future__ import annotations

import importlib

#: Each step's function and the module defining it.
_STEPS = {
    "setup": "setup",
    "arena_tool": "arena",
    "extract": "extract",
    "diagnostics": "diagnostics",
    "train_pca": "pca",
    "scores_and_changepoints": "scores",
}

__all__ = list(_STEPS)


def __getattr__(name: str):
    if name not in _STEPS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    function = getattr(importlib.import_module(f"{__name__}.{_STEPS[name]}"), name)
    # importing the module bound its name here; a step's function and module may share it
    globals()[name] = function
    return function

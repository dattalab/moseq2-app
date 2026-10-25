"""Reach the widgets moseq2-app builds.

Most of the notebook's interactive entry points hand their widgets to
IPython's ``display()`` and return something else: ``interactive_syllable_stats``
and ``show_dendrogram`` return ``None``, ``label_syllables`` returns an integer.
The widget objects are otherwise unreachable from a test.

``capture_display`` replaces ``display`` in every module that imported it with
a recorder that keeps each argument and returns ``None``, which is what the
real ``display`` returns outside a kernel.
"""

from __future__ import annotations

import importlib

#: Every module that does ``from IPython.display import display``. The name is
#: bound at import time, so patching ``IPython.display`` is not enough: each
#: module holds its own reference to the function.
DISPLAY_IMPORTERS = (
    "moseq2_app.main",
    "moseq2_app.gui.wrappers",
    "moseq2_app.viz.view",
    "moseq2_app.viz.controller",
    "moseq2_app.roi.view",
)


class DisplayRecorder:
    """Stands in for ``display`` and remembers what it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, *objects, **kwargs):
        self.calls.append(objects)
        return None

    @property
    def objects(self):
        """Every displayed object, flattened across calls, in order."""
        return [obj for call in self.calls for obj in call]

    def of_type(self, cls):
        return [obj for obj in self.objects if isinstance(obj, cls)]

    def one(self, cls):
        """The single displayed object of ``cls``, or a failure naming what was there."""
        found = self.of_type(cls)
        if len(found) != 1:
            seen = [type(obj).__name__ for obj in self.objects]
            raise AssertionError(
                f"expected exactly one displayed {cls.__name__}, found {len(found)}. "
                f"Displayed: {seen}"
            )
        return found[0]

    def clear(self):
        self.calls.clear()


def capture_display(monkeypatch, modules=DISPLAY_IMPORTERS):
    """Patch ``display`` in every module that imported it; return the recorder.

    ``monkeypatch`` undoes the patch at the end of the test that asked for it.
    """
    recorder = DisplayRecorder()
    patched = []
    for name in modules:
        module = importlib.import_module(name)
        if hasattr(module, "display"):
            monkeypatch.setattr(module, "display", recorder)
            patched.append(name)
    if not patched:
        raise RuntimeError(
            "no module in DISPLAY_IMPORTERS exposed `display`; the import sites moved"
        )
    return recorder


def layout_of(view):
    """The Panel layout a ``Viewer`` renders as."""
    import panel as pn

    if isinstance(view, pn.viewable.Viewer):
        return view.__panel__()
    return view


def buttons_of(view):
    """Every button in a layout, indexed by the label it shows the user."""
    import panel as pn

    return {b.name: b for b in layout_of(view).select(pn.widgets.Button)}


def click(button):
    """Press a button the way the browser does.

    moseq2-app builds its buttons from ``param.Action`` parameters with
    ``Button.from_param``, which links the button's ``value`` to the action.
    Assigning ``value`` runs the callback. Bumping ``clicks``, which is the
    count the browser keeps, does not run anything, so a test written that way
    passes while pressing nothing.
    """
    button.value = True

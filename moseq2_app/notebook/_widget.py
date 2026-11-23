"""What every step's GUI shares: finding its project, showing a problem, and running long work.

``StepWidget`` is a step's GUI: an anywidget whose page is ``static/kit.js``
(the controls and the problem panel) and the step's own script, styled by
``static/kit.css``. It finds its project in the kernel's working folder, where
the notebook sits (``Project.at(Path.cwd())``), and shows a problem panel
(``problem``) when that folder holds no ``config.yaml``. An expected error
(``moseq2_common.errors.ExpectedError``) is shown as its message, its fix and
its help page; anything else as a panel saying something unexpected went wrong,
with the traceback and what the step printed folded under "Details for a bug
report". Nothing a step prints reaches the cell: ``moseq2_common.capture``
keeps it.

``RunningStep`` is a step that does long work. Its run button sends
``{"action": "start", "settings": {...}}``; ``start`` saves nothing itself and
returns at once, running the step's work in a background thread
(``moseq2_common.capture.in_background``) with a ``GuiReporter``, so the notebook
stays usable while it runs. ``stop`` asks the work to stop at its next chunk;
``wait`` blocks until the run ends. The ``run`` trait is what the page draws:
``{"state", "where", "items", "finished_at"}``, ``state`` one of ``idle``,
``running``, ``finished``, ``failed``, ``stopped``.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import ClassVar

import anywidget
import traitlets

from moseq2_common import capture
from moseq2_common.errors import ExpectedError, StepStopped, details_for_bug
from moseq2_common.project import Project
from moseq2_common.report import reporting

__all__ = ["GuiReporter", "RunningStep", "StepWidget", "problem_of", "static_text"]

#: The bug panel's title and text (W6); its traceback is folded under "Details for a bug report".
BUG_TITLE = "Something unexpected went wrong"
BUG_TEXT = ("This looks like a bug in MoSeq2, not a problem with your data or settings. Please report it at "
            "https://github.com/dattalab/moseq2-app/issues and include the details below.")


def static_text(*names: str) -> str:
    """The files ``names`` of ``static/``, joined in order."""
    folder = resources.files("moseq2_app.notebook") / "static"
    return "\n".join((folder / name).read_text(encoding="utf-8") for name in names)


def problem_of(error: BaseException, details: str = "") -> dict:
    """The problem panel for ``error``: an expected error's message, fix and help page, or the bug panel."""
    if isinstance(error, ExpectedError):
        return {"kind": "expected", "title": error.title or error.message,
                "text": error.message if error.title else "", "fix": error.fix or "", "help": error.help_page or ""}
    return {"kind": "bug", "title": BUG_TITLE, "text": BUG_TEXT, "details": details_for_bug(error, details)}


def no_project(folder: Path) -> ExpectedError:
    """What a step's GUI says when the notebook is not in a project folder (W7)."""
    return ExpectedError(f"This notebook isn't in a MoSeq2 project folder yet: there's no config.yaml in {folder}.",
                         fix="run Setup (step 1) and press Save, or move this notebook into your project folder "
                             "and open it from there.")


class StepWidget(anywidget.AnyWidget):
    """A step's GUI in the notebook (module docstring)."""

    #: The step, as ``moseq2_common.project.step_lock`` names it.
    STEP: ClassVar[str] = ""
    #: The pipeline phase the step belongs to, for its colour: ``extraction``, ``pca``, ``modeling``, ``analysis``.
    PHASE: ClassVar[str] = ""

    phase = traitlets.Unicode("").tag(sync=True)
    problem = traitlets.Dict().tag(sync=True)
    lines = traitlets.List().tag(sync=True)
    _css = static_text("kit.css")

    def __init__(self, project: Project | None = None, **kwargs):
        super().__init__(phase=self.PHASE, **kwargs)
        capture.install()
        self.project = project if project is not None else Project.at(Path.cwd())
        self.on_msg(self._on_msg)
        from moseq2_app.notebook import _state

        _state.opened(self)

    def has_project(self) -> bool:
        """Whether the project has its ``config.yaml``; when not, the problem panel says so (and stops saying
        so once it has one)."""
        missing = problem_of(no_project(Path(self.project.folder)))
        if Path(self.project.config_file).is_file():
            if self.problem == missing:
                self.problem = {}
            return True
        self.problem = missing
        return False

    def report_problem(self, error: BaseException, details: str = "") -> None:
        """Show ``error`` as the GUI's problem panel."""
        self.problem = problem_of(error, details)

    def refresh(self) -> None:
        """Read the project again (another GUI saved, or a run ended). What reading it prints is kept out of the
        cell, a warning is kept in ``lines``, and a problem is shown in the panel."""
        if not self.has_project():
            return
        reporter = _LineReporter(self)
        try:
            with reporting(reporter), capture.captured(reporter, io.StringIO()):
                self._read()
        except Exception as error:  # noqa: BLE001 -- shown in the GUI, never as a traceback in the cell
            self.report_problem(error)

    def _read(self) -> None:
        """Read what the GUI shows from the project; each step's own."""

    def _line(self, kind: str, text: str) -> None:
        self.lines = [*self.lines[-199:], {"kind": kind, "text": text}]

    def _on_msg(self, _widget, content, _buffers) -> None:
        handler = getattr(self, f"_on_{content.get('action')}", None)
        if handler is not None:
            try:
                handler(content)
            except Exception as error:  # noqa: BLE001 -- shown in the GUI, never as a traceback in the cell
                self.report_problem(error)


class _LineReporter:
    """Where reading a project reports outside a run: its status lines and warnings kept in ``lines``."""

    def __init__(self, widget: StepWidget):
        self._widget = widget

    def status(self, text: str) -> None:
        self._widget._line("status", text)

    def warning(self, text: str) -> None:
        self._widget._line("warning", text)

    def progress(self, label: str, total: int | None = None, *, unit: str = ""):
        return _NoBar()

    def stop_requested(self) -> bool:
        return False


class _Row:
    """The progress bar of one item's row: a run's progress labelled with the item's key."""

    def __init__(self, reporter: GuiReporter, key: str, total: int | None):
        self._reporter, self._key = reporter, key
        reporter.update(key, state="running", done=0, total=total)

    def advance(self, n: int = 1) -> None:
        self._reporter.advance(self._key, n)

    def __enter__(self) -> _Row:
        return self

    def __exit__(self, kind, error, _traceback) -> None:
        if error is None or isinstance(error, GeneratorExit):
            self._reporter.update(self._key, state="done")
        elif isinstance(error, StepStopped):
            self._reporter.update(self._key, state="stopped")
        elif isinstance(error, Exception):
            self._reporter.update(self._key, state="failed", problem=problem_of(error))


class _NoBar:
    def advance(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> _NoBar:
        return self

    def __exit__(self, *exc) -> None:
        pass


class GuiReporter:
    """A step's reporter in its GUI: a progress labelled with one of the run's item keys drives that item's
    row; other bars are not drawn. Status lines and warnings are kept in the widget's ``lines``."""

    def __init__(self, widget: RunningStep, keys: Sequence[str] = ()):
        self._widget = widget
        self._keys = set(keys)

    def status(self, text: str) -> None:
        self._widget._line("status", text)

    def warning(self, text: str) -> None:
        self._widget._line("warning", text)

    def progress(self, label: str, total: int | None = None, *, unit: str = ""):
        if label in self._keys:
            return _Row(self, label, total)
        return _NoBar()

    def stop_requested(self) -> bool:
        return self._widget._stopping.is_set()

    def update(self, key: str, **values) -> None:
        self._widget._update_item(key, values)

    def advance(self, key: str, n: int) -> None:
        self._widget._advance_item(key, n)


class RunningStep(StepWidget):
    """A step that does long work in the background (module docstring)."""

    sections = traitlets.List().tag(sync=True)
    saved = traitlets.Dict().tag(sync=True)
    status = traitlets.Dict().tag(sync=True)
    results = traitlets.Dict().tag(sync=True)
    run = traitlets.Dict().tag(sync=True)

    #: How many status lines and warnings ``lines`` keeps.
    KEPT_LINES = 200

    def __init__(self, project: Project | None = None, **kwargs):
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._finished = threading.Event()
        self._finished.set()
        self._thread = None
        super().__init__(project, run={"state": "idle", "items": {}}, **kwargs)

    # ---- what a step gives
    def _work(self, settings: Mapping, reporter: GuiReporter):
        """Run the step with ``settings`` (as the page sends them), reporting to ``reporter``; its result."""
        raise NotImplementedError

    def _items(self) -> list[str]:
        """The keys of the rows a run draws."""
        return []

    def _where(self, settings: Mapping) -> str:
        return "this computer"

    def _finish(self, outcome: capture.Outcome) -> dict:
        """The final ``run`` dict for ``outcome``."""
        with self._lock:
            run = dict(self.run)
        error = outcome.error
        if error is None:
            run["state"] = "finished"
        elif isinstance(error, StepStopped):
            run["state"] = "stopped"
        else:
            run["state"] = "failed"
            self.report_problem(error, outcome.details)
        run["finished_at"] = time.strftime("%H:%M")
        return run

    # ---- running
    def start(self, settings: Mapping | None = None) -> None:
        """Start the step's work in the background with ``settings`` (as the run button sends them); returns at
        once. A start while a run is going is ignored."""
        with self._lock:
            if not self._finished.is_set():
                return
            if not self.has_project():
                return
            self.problem = {}
            self.lines = []
            self._stopping.clear()
            self._finished.clear()
            keys = self._items()
            settings = dict(settings or {})
            self.run = {"state": "running", "where": self._where(settings),
                        "items": {key: {"state": "waiting", "done": 0, "total": None, "problem": {}} for key in keys}}
        reporter = GuiReporter(self, keys)
        self._thread = capture.in_background(lambda: self._work(settings, reporter), reporter=reporter,
                                             on_finish=self._ended)

    def stop(self) -> None:
        """Ask the run to stop at its next chunk of work."""
        self._stopping.set()

    def wait(self, timeout: float | None = None) -> dict:
        """Wait for the run to end (or ``timeout`` seconds); the ``run`` dict."""
        self._finished.wait(timeout)
        return dict(self.run)

    def _ended(self, outcome: capture.Outcome) -> None:
        try:
            run = self._finish(outcome)
        except Exception as error:  # noqa: BLE001 -- shown in the GUI
            run = {**self.run, "state": "failed"}
            self.report_problem(error, outcome.details)
        with self._lock:
            self.run = run
        from moseq2_app.notebook import _state

        _state.changed(self.project)
        self._finished.set()

    def _on_start(self, content: Mapping) -> None:
        self.start(content.get("settings"))

    def _on_stop(self, _content: Mapping) -> None:
        self.stop()

    # ---- the rows, from the reporter
    def _line(self, kind: str, text: str) -> None:
        with self._lock:
            self.lines = [*self.lines[-(self.KEPT_LINES - 1):], {"kind": kind, "text": text}]

    def _update_item(self, key: str, values: Mapping) -> None:
        with self._lock:
            items = {k: dict(v) for k, v in self.run.get("items", {}).items()}
            item = items.setdefault(key, {"state": "waiting", "done": 0, "total": None, "problem": {}})
            item.update(values)
            if values.get("state") == "done" and item.get("total"):
                item["done"] = item["total"]
            self.run = {**self.run, "items": items}

    def _advance_item(self, key: str, n: int) -> None:
        with self._lock:
            item = self.run.get("items", {}).get(key, {})
            self._update_item(key, {"done": item.get("done", 0) + n})

"""How a step tells the person what it is doing: status lines, progress and warnings.

Every step function takes ``reporter: Reporter | None = None`` and runs its body
inside ``reporting(reporter)``, which makes that reporter ``current()`` for the
rest of the step in this thread (a ``contextvars.ContextVar``). Code below the
step function reports through ``current()`` rather than a parameter:

- ``current().status(text)``: a line saying what is happening and why;
- ``current().warning(text)``: something the person should know, which does not
  stop the step;
- ``with current().progress(label, total) as bar: bar.advance()``: a bar;
  ``progress(label, total, show=...)`` and ``iterate(items, label, show=...)``
  are the same through the current reporter, drawing nothing when ``show`` is
  False (a bar a setting such as ``progress_bar`` turns on).

``reporting(None)`` keeps the reporter already current, so a step called by
another step reports to its caller's reporter. Outside any ``reporting`` block,
``current()`` is the default reporter (``default_reporter``): a
``TerminalReporter`` writing status lines, ``Warning: ``-prefixed warnings and
tqdm bars to ``sys.stderr``, so a script gets what a GUI shows and its standard
output stays its own. The command line passes a ``TerminalReporter`` honouring
``--quiet``; a notebook GUI passes one that draws its rows and status lines.

A long loop calls ``checkpoint()`` once per chunk of work: it raises
``moseq2_common.errors.StepStopped`` when the current reporter says the person
asked to stop (``stop_requested``). Python ``logging`` (loggers named
``moseq2.<package>``) carries debugging detail only.

A step running where no one watches it, such as a cluster job, reports through
a ``ProgressFileReporter``: a small JSON file replaced whole every couple of
seconds (``{"state", "stage", "done", "total", "updated", "message", "fix",
"job"}``), which whoever follows the step reads with ``read_progress``.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import sys
import threading
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO, runtime_checkable

from moseq2_common.errors import ExpectedError, StepStopped

__all__ = ["Progress", "ProgressFileReporter", "Reporter", "TerminalReporter", "checkpoint", "current",
           "default_reporter", "iterate", "progress", "read_progress", "reporting"]


@runtime_checkable
class Progress(Protocol):
    """A progress bar, used as a context manager; ``advance`` moves it on."""

    def advance(self, n: int = 1) -> None: ...

    def __enter__(self) -> Progress: ...

    def __exit__(self, *exc) -> None: ...


@runtime_checkable
class Reporter(Protocol):
    """Where a step's status lines, warnings and progress go."""

    def status(self, text: str) -> None: ...

    def warning(self, text: str) -> None: ...

    def progress(self, label: str, total: int | None = None, *, unit: str = "") -> Progress: ...

    def stop_requested(self) -> bool: ...


class _NoProgress:
    """A bar that draws nothing."""

    def advance(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> _NoProgress:
        return self

    def __exit__(self, *exc) -> None:
        pass


class _TqdmProgress:
    """A tqdm bar on a stream."""

    def __init__(self, label: str, total: int | None, unit: str, stream: TextIO):
        from tqdm.auto import tqdm

        self._bar = tqdm(total=total, desc=label, unit=unit or "it", file=stream)

    def advance(self, n: int = 1) -> None:
        self._bar.update(n)

    def __enter__(self) -> _TqdmProgress:
        return self

    def __exit__(self, *exc) -> None:
        self._bar.close()


class TerminalReporter:
    """Status lines, ``Warning: ``-prefixed warnings and tqdm bars on ``stream`` (``sys.stderr`` when None).

    With ``quiet``, only warnings are written. ``sys.stderr`` is looked up at each write, so a stream an
    interface swaps in later (``moseq2_common.capture``) is the one written to.
    """

    def __init__(self, stream: TextIO | None = None, *, quiet: bool = False):
        self._stream = stream
        self.quiet = quiet

    @property
    def stream(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stderr

    def status(self, text: str) -> None:
        if not self.quiet:
            print(text, file=self.stream, flush=True)

    def warning(self, text: str) -> None:
        print(f"Warning: {text}", file=self.stream, flush=True)

    def progress(self, label: str, total: int | None = None, *, unit: str = "") -> Progress:
        if self.quiet:
            return _NoProgress()
        return _TqdmProgress(label, total, unit, self.stream)

    def stop_requested(self) -> bool:
        return False


_DEFAULT = TerminalReporter()
_CURRENT: contextvars.ContextVar[Reporter | None] = contextvars.ContextVar("moseq2_reporter", default=None)


def default_reporter() -> Reporter:
    """The reporter a step uses when it is given none: a ``TerminalReporter`` on ``sys.stderr``."""
    return _DEFAULT


def current() -> Reporter:
    """The reporter of the step running in this thread; ``default_reporter()`` outside any step."""
    reporter = _CURRENT.get()
    return reporter if reporter is not None else default_reporter()


@contextlib.contextmanager
def reporting(reporter: Reporter | None) -> Iterator[Reporter]:
    """Make ``reporter`` current for the block; None keeps the reporter already current."""
    if reporter is None:
        yield current()
        return
    token = _CURRENT.set(reporter)
    try:
        yield reporter
    finally:
        _CURRENT.reset(token)


def checkpoint() -> None:
    """Raise ``StepStopped`` when the person asked the running step to stop (``Reporter.stop_requested``)."""
    if current().stop_requested():
        raise StepStopped("Stopped.")


def progress(label: str, total: int | None = None, *, unit: str = "", show: bool = True) -> Progress:
    """``current().progress(label, total)``; a bar that draws nothing when ``show`` is False."""
    if not show:
        return _NoProgress()
    return current().progress(label, total, unit=unit)


def iterate(items: Iterable, label: str, *, total: int | None = None, unit: str = "", show: bool = True) -> Iterator:
    """``items``, one at a time, with a bar ``label`` advanced as each is done (``progress``)."""
    if total is None and hasattr(items, "__len__"):
        total = len(items)
    with progress(label, total, unit=unit, show=show) as bar:
        for item in items:
            yield item
            bar.advance()


# ------------------------------------------------------------------------------- a progress file


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class _FileBar:
    """A bar of a ``ProgressFileReporter``: counted in the file when it is the bar the file follows, and passed on
    to the reporter's ``also`` as it goes."""

    def __init__(self, owner: ProgressFileReporter, label: str, total: int | None, unit: str):
        self._owner = owner
        self._counted = owner._follows(label)
        self._also = owner._also.progress(label, total, unit=unit) if owner._also is not None else _NoProgress()
        owner._set(stage=label, **({"done": 0, "total": total} if self._counted else {}))

    def advance(self, n: int = 1) -> None:
        if self._counted:
            self._owner._advance(n)
        self._also.advance(n)

    def __enter__(self) -> _FileBar:
        self._also.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._also.__exit__(*exc)


class ProgressFileReporter:
    """A reporter that keeps a step's progress in the JSON file ``path``, for whoever follows the step from elsewhere.

    Used as a context manager around the step: on entering, the file says ``running``; while the step runs it is
    rewritten whole (through a temporary name) every ``every`` seconds; on leaving it says ``finished``, or
    ``failed`` with an expected error's text as ``message`` and its fix as ``fix``. ``stage`` is the last status
    line or bar label; ``done`` and ``total`` count the bar labelled ``bar`` (every bar when None);
    ``updated`` is when the file was written, with its UTC offset; ``job`` is the SLURM job running it
    (``SLURM_JOB_ID``), or None. Every call is passed on to ``also`` as well, when given (a job's log).
    """

    def __init__(self, path: str | os.PathLike, *, every: float = 2.0, bar: str | None = None,
                 also: Reporter | None = None):
        self.path = Path(path)
        self.every = every
        self._bar = bar
        self._also = also
        self._lock = threading.Lock()
        self._state = {"state": "running", "stage": "", "done": 0, "total": None, "updated": None, "message": None,
                       "fix": None, "job": os.environ.get("SLURM_JOB_ID")}
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- the reporter
    def status(self, text: str) -> None:
        self._set(stage=text)
        if self._also is not None:
            self._also.status(text)

    def warning(self, text: str) -> None:
        if self._also is not None:
            self._also.warning(text)

    def progress(self, label: str, total: int | None = None, *, unit: str = "") -> Progress:
        return _FileBar(self, label, total, unit)

    def stop_requested(self) -> bool:
        return self._also.stop_requested() if self._also is not None else False

    # ---- the file
    def _follows(self, label: str) -> bool:
        return self._bar is None or label == self._bar

    def _set(self, **values) -> None:
        with self._lock:
            self._state.update(values)

    def _advance(self, n: int) -> None:
        with self._lock:
            self._state["done"] += n

    def write(self) -> None:
        """Write the file now, whole."""
        with self._lock:
            state = {**self._state, "updated": _now()}
        partial = self.path.with_name(f".{self.path.name}.{os.getpid()}.partial")
        partial.write_text(json.dumps(state))
        os.replace(partial, self.path)

    def _beat(self) -> None:
        while not self._stopped.wait(self.every):
            try:
                self.write()
            except OSError:
                pass  # the next beat tries again; a step never fails for its progress file

    def __enter__(self) -> ProgressFileReporter:
        self.write()
        self._stopped.clear()
        self._thread = threading.Thread(target=self._beat, name="moseq2-progress-file", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, kind, error, _traceback) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()
        if error is None:
            with self._lock:
                total = self._state["total"]
                if total is not None:
                    self._state["done"] = total
            self._set(state="finished")
        elif isinstance(error, ExpectedError):
            self._set(state="failed", message=str(error), fix=error.fix)
        else:
            self._set(state="failed")
        self.write()


def read_progress(path: str | os.PathLike) -> dict | None:
    """The progress file ``path`` as a ``ProgressFileReporter`` wrote it; None for a missing, empty, half-written
    or otherwise unreadable file."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "state" in data else None

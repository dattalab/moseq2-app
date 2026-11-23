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
"""

from __future__ import annotations

import contextlib
import contextvars
import sys
from collections.abc import Iterable, Iterator
from typing import Protocol, TextIO, runtime_checkable

from moseq2_common.errors import StepStopped

__all__ = ["Progress", "Reporter", "TerminalReporter", "checkpoint", "current", "default_reporter", "iterate",
           "progress", "reporting"]


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

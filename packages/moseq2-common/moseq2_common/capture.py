"""Catching everything a running step emits, so it reaches its interface and nothing else.

The layers, and the libraries under them, print, warn, log and draw progress
bars. An interface decides where each goes: the command line shows status lines
and warnings and keeps the rest for a bug report; a notebook GUI shows them in
its own rows and never beside itself, even while the step runs in a background
thread and the person runs other cells.

``install()`` (idempotent; every interface calls it) replaces ``sys.stdout`` and
``sys.stderr`` with proxies, wraps ``warnings.showwarning`` and adds one root
``logging`` handler. Each consults a ``contextvars.ContextVar`` holding the
current run's sink and, when there is none, passes through untouched: a cell's
own ``print`` reaches the cell as before. Inside ``captured(reporter, details)``:

- text written to ``sys.stdout`` or ``sys.stderr`` (``print``, a tqdm bar a
  library draws) goes to ``details``;
- log records go to ``details``;
- warnings go to ``reporter.warning``.

The sink lives in the context of the code that set it, so a step run in another
thread under ``in_background`` captures only its own output: a print from the
cell running meanwhile reaches that cell. ``terminal()`` is the stderr beneath
the proxy, where the command line's reporter writes.

Output written below Python, to the process's file descriptors 1 and 2, is out
of reach of the proxies, so a subprocess the layers start gets pipes, never the
process's own descriptors.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import logging
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TextIO

from moseq2_common.report import Reporter, reporting

__all__ = ["Outcome", "captured", "in_background", "install", "terminal"]


@dataclass(frozen=True)
class _Sink:
    reporter: Reporter
    details: TextIO


_SINK: contextvars.ContextVar[_Sink | None] = contextvars.ContextVar("moseq2_capture", default=None)
_LOCK = threading.Lock()


class _Proxy(io.TextIOBase):
    """Stands in for ``sys.stdout`` or ``sys.stderr``: writes go to the current run's details, or, outside a
    run, to the stream it replaced."""

    def __init__(self, original: TextIO):
        super().__init__()
        self.original = original

    def write(self, text: str) -> int:
        sink = _SINK.get()
        if sink is None:
            return self.original.write(text)
        sink.details.write(text)
        return len(text)

    def flush(self) -> None:
        if _SINK.get() is None:
            self.original.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return _SINK.get() is None and self.original.isatty()

    def fileno(self) -> int:
        return self.original.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self.original, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.original, name)


def _show_warning(message, category, filename, lineno, file=None, line=None):
    sink = _SINK.get()
    if sink is None:
        return _show_warning.previous(message, category, filename, lineno, file, line)
    sink.reporter.warning(str(message))


class _Handler(logging.Handler):
    """The root handler: a record goes to the current run's details; outside a run, it is left to the other
    handlers, or to logging's last resort when there are none, as before this handler was added."""

    def emit(self, record: logging.LogRecord) -> None:
        sink = _SINK.get()
        if sink is not None:
            sink.details.write(self.format(record) + "\n")
            return
        if not self._handled_elsewhere(record) and logging.lastResort is not None \
                and record.levelno >= logging.lastResort.level:
            logging.lastResort.handle(record)

    def _handled_elsewhere(self, record: logging.LogRecord) -> bool:
        logger = logging.getLogger(record.name)
        while logger is not None:
            if any(handler is not self for handler in logger.handlers):
                return True
            if not logger.propagate:
                break
            logger = logger.parent
        return False


_HANDLER = _Handler()
_HANDLER.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))


def install() -> None:
    """Put the proxies, the warning hook and the logging handler in place; anything already in place is kept."""
    with _LOCK:
        if not isinstance(sys.stdout, _Proxy):
            sys.stdout = _Proxy(sys.stdout)
        if not isinstance(sys.stderr, _Proxy):
            sys.stderr = _Proxy(sys.stderr)
        if warnings.showwarning is not _show_warning:
            _show_warning.previous = warnings.showwarning
            warnings.showwarning = _show_warning
        root = logging.getLogger()
        if _HANDLER not in root.handlers:
            root.addHandler(_HANDLER)


def terminal() -> TextIO:
    """The standard error beneath the proxy: where output meant for a person at a terminal goes."""
    stream = sys.stderr
    while isinstance(stream, _Proxy):
        stream = stream.original
    return stream


@contextlib.contextmanager
def captured(reporter: Reporter, details: TextIO) -> Iterator[None]:
    """For the block, in this context: printed text and log records go to ``details``, warnings to
    ``reporter.warning``. ``install()`` must have run."""
    token = _SINK.set(_Sink(reporter, details))
    try:
        yield
    finally:
        _SINK.reset(token)


@dataclass(frozen=True)
class Outcome:
    """How a background step ended: its return ``value``, or the ``error`` it raised, and what it printed."""

    value: object
    error: BaseException | None
    details: str


def in_background(function: Callable[[], object], *, reporter: Reporter,
                  on_finish: Callable[[Outcome], None]) -> threading.Thread:
    """Run ``function()`` in a new thread with ``reporter`` current and everything it emits captured; call
    ``on_finish`` with its ``Outcome`` when it ends. Returns the started thread."""
    install()

    def run():
        details = io.StringIO()
        value, error = None, None
        with reporting(reporter), captured(reporter, details):
            try:
                value = function()
            except BaseException as raised:  # noqa: BLE001 -- the interface shows it
                error = raised
        on_finish(Outcome(value=value, error=error, details=details.getvalue()))

    context = contextvars.copy_context()
    thread = threading.Thread(target=context.run, args=(run,), daemon=True, name="moseq2-step")
    thread.start()
    return thread

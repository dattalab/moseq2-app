"""Errors a step expects, which every interface can tell from a bug.

An ``ExpectedError`` is a problem with the person's data, settings or project
that a step found and can explain: a missing input, a bad setting, a stale
record, a stop the step chose. It carries its ``message`` (what was found), and
optionally a ``fix`` (what to do), a ``help_page`` (the title of the help page
that explains more), a ``title`` (a problem panel's heading, when the message is
its text) and a ``command_line_fix`` (the fix in the command line's words, where
the GUI's names a button or a cell). The step classes that report such problems derive from
it and keep their other bases (``ValueError``, ``FileNotFoundError`` and the
like), so code catching those still catches them.

The interfaces show an expected error and a bug differently. On the command line
an expected error prints ``Error: <message>`` and, when it has one,
``To fix it: <fix>``, and the command exits 1; anything else prints its
traceback and the captured output (``details_for_bug``) with a line asking for a
bug report, and exits 3. In the notebook an expected error is a problem panel;
anything else is a panel saying something unexpected went wrong, with the
traceback and captured output folded under "Details for a bug report".
"""

from __future__ import annotations

import traceback

__all__ = ["ExpectedError", "StepRunning", "StepStopped", "details_for_bug"]


class ExpectedError(Exception):
    """A problem a step found and can explain; ``str()`` of it is ``message``, after ``title`` when it has one."""

    def __init__(self, message: str = "", *, fix: str | None = None, help_page: str | None = None,
                 title: str | None = None, command_line_fix: str | None = None):
        super().__init__(message)
        self.message = message
        self.fix = fix
        self.help_page = help_page
        self.title = title
        self.command_line_fix = command_line_fix

    def __str__(self) -> str:
        return f"{self.title}. {self.message}" if self.title else self.message


class StepStopped(ExpectedError):
    """A step stopped because the person asked it to (``moseq2_common.report.checkpoint``)."""


class StepRunning(ExpectedError):
    """A step is already running in this project (``moseq2_common.project.step_lock``)."""


def details_for_bug(error: BaseException, captured: str = "") -> str:
    """What a bug report needs: ``error``'s traceback, then the output captured while the step ran."""
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    if captured:
        text += "\nOutput captured while it ran:\n" + captured
    return text

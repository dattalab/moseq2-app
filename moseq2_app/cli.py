"""The ``moseq2`` command: every pipeline step in a project, and ``moseq2 standalone`` for files named.

``moseq2 <step>`` runs a step in the project in the working folder through its
project layer (``moseq2_<package>.project``), the same function with the same
settings the notebook's step calls. ``moseq2 standalone <step>`` runs a step's
file layer (``moseq2_<package>.steps``) on the files named on the command line,
reading no project; it also holds the utilities that are not pipeline steps.
``python -m moseq2_app`` is the same command.

The commands live in the modules of ``moseq2_app.commands``, each declaring its
``ORDER`` and its ``STEPS``, ``STANDALONE``, ``RECORDS`` and ``HIDDEN`` commands;
``cli`` and ``standalone`` are built from them, so a module adds commands
without this one changing. ``moseq2 --help`` lists them under three headings, in
the modules' order.

How a run ends (``run_step``): an expected error
(``moseq2_common.errors.ExpectedError``) prints ``Error: <message>`` and, when it
has one, ``To fix it: <fix>`` on standard error and exits 1; a usage error exits
2; anything else prints its traceback, what the step printed while it ran, and a
line asking for a bug report, and exits 3. ``--debug`` shows tracebacks of
expected errors too, and what the step prints as it prints it; ``--quiet``
keeps warnings and errors only. Both are options of every command
(``common_options``). Status lines, warnings and progress bars go to standard
error through a ``moseq2_common.report.TerminalReporter``; everything else the
step and its libraries print is captured (``moseq2_common.capture``) and shown
only with ``--debug`` or for a bug.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import click
from click.core import ParameterSource

from moseq2_common import capture
from moseq2_common.errors import ExpectedError
from moseq2_common.options import Options, _spec
from moseq2_common.project import Project
from moseq2_common.report import Reporter, TerminalReporter, reporting

__all__ = ["EXIT_BUG", "EXIT_EXPECTED", "EXIT_USAGE", "cli", "common_options", "project_here", "run_step",
           "settings_flags", "standalone", "typed_settings"]

T = TypeVar("T")

EXIT_EXPECTED, EXIT_USAGE, EXIT_BUG = 1, 2, 3

#: The line a bug ends with on the command line.
BUG_LINE = ("This looks like a bug in MoSeq2; please report it at https://github.com/dattalab/moseq2-app/issues "
            "with the traceback above.")

#: The headings ``moseq2 --help`` lists its commands under.
STEPS_HEADING = "Steps, in your project"
STANDALONE_HEADING = "Standalone: any step on files you name, for scripts"
RECORDS_HEADING = "Records"


# ------------------------------------------------------------------------------- running a step


class _Tee(io.TextIOBase):
    """Keeps what is written and passes it on to ``stream`` as it comes (``--debug``)."""

    def __init__(self, stream):
        super().__init__()
        self.kept = io.StringIO()
        self._stream = stream

    def write(self, text: str) -> int:
        self.kept.write(text)
        self._stream.write(text)
        return len(text)

    def getvalue(self) -> str:
        return self.kept.getvalue()


def run_step(work: Callable[[Reporter], T], *, quiet: bool = False, debug: bool = False) -> T:
    """Run ``work(reporter)`` as the command line runs a step, ending as the module docstring says."""
    capture.install()
    terminal = capture.terminal()
    reporter = TerminalReporter(terminal, quiet=quiet)
    details = _Tee(terminal) if debug else io.StringIO()
    logger = logging.getLogger("moseq2")
    level = logger.level
    if debug:
        logger.setLevel(logging.DEBUG)
    try:
        with reporting(reporter), capture.captured(reporter, details):
            return work(reporter)
    except (click.exceptions.Exit, click.ClickException, click.Abort, KeyboardInterrupt):
        raise
    except ExpectedError as error:
        if debug:
            traceback.print_exception(type(error), error, error.__traceback__, file=terminal)
        click.echo(f"Error: {error}", file=terminal)
        fix = error.command_line_fix or error.fix
        if fix:
            click.echo(f"To fix it: {fix}", file=terminal)
        raise click.exceptions.Exit(EXIT_EXPECTED) from None
    except Exception as error:  # noqa: BLE001 -- a bug: shown with everything a report needs
        traceback.print_exception(type(error), error, error.__traceback__, file=terminal)
        captured = details.getvalue()
        if captured and not debug:
            click.echo("\nOutput captured while it ran:", file=terminal)
            click.echo(captured, file=terminal, nl=not captured.endswith("\n"))
        click.echo(BUG_LINE, file=terminal)
        raise click.exceptions.Exit(EXIT_BUG) from None
    finally:
        logger.setLevel(level)


def common_options(command):
    """``--quiet``/``-q`` and ``--debug``, for every command; the command takes ``quiet`` and ``debug``."""
    command = click.option("--debug", is_flag=True, default=False,
                           help="Show the full traceback of any error, and everything the step prints as it runs.")(
        command)
    command = click.option("--quiet", "-q", is_flag=True, default=False,
                           help="Show only warnings and errors.")(command)
    return command


def project_here(command: str) -> Project:
    """The project in the working folder; ``ExpectedError`` (W7) when the folder holds no ``config.yaml``."""
    folder = Path.cwd()
    project = Project.at(folder)
    if not Path(project.config_file).is_file():
        raise ExpectedError(f"`moseq2 {command}` runs in a project folder, and {folder} has no config.yaml.",
                            fix="run it from your project folder, or start a project here with `moseq2 setup`.")
    return project


def settings_flags(*classes: type[Options], names: Sequence[str] | None = None):
    """The click options of the settings of ``classes`` (in their declaration order, each name once), or of
    ``names`` only, in that order, each with its declared flags, type and help. What was typed is read with
    ``typed_settings``."""
    fields: dict = {}
    for cls in classes:
        for field in dataclasses.fields(cls):
            fields.setdefault(field.name, field)
    chosen = list(names) if names is not None else list(fields)

    def apply(function):
        for name in reversed(chosen):
            spec = _spec(fields[name])
            function = click.option(name, *spec["flags"], type=spec["type"], multiple=spec["multiple"],
                                    is_flag=spec["is_flag"], default=fields[name].default, show_default=False,
                                    help=spec["help"])(function)
        return function

    return apply


def typed_settings(ctx: click.Context, classes: Sequence[type[Options]]) -> dict:
    """The settings of ``classes`` typed on ``ctx``'s command line, with their values."""
    names = {f.name for cls in classes for f in dataclasses.fields(cls)}
    return {name: value for name, value in ctx.params.items()
            if name in names and ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE}


# ------------------------------------------------------------------------------- the groups


def _modules():
    from moseq2_app.commands import modules

    return modules()


def _commands(kind: str) -> list[click.Command]:
    return [command for module in _modules() for command in getattr(module, kind, ())]


class _Moseq2(click.Group):
    """``moseq2``: its help lists the steps, then each standalone command, then the records commands."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        limit = formatter.width - 6 - 24
        sections = (
            (STEPS_HEADING, [(c.name, c) for c in _commands("STEPS")]),
            (STANDALONE_HEADING, [(f"standalone {c.name}", c) for c in _commands("STANDALONE")]),
            (RECORDS_HEADING, [(c.name, c) for c in _commands("RECORDS")]),
        )
        for heading, rows in sections:
            if rows:
                with formatter.section(heading):
                    formatter.write_dl([(name, command.get_short_help_str(limit)) for name, command in rows])


@click.group(cls=_Moseq2, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="moseq2-app", prog_name="moseq2")
def cli():
    """MoSeq2: from depth recordings to behavioral syllables.

    Run a step from your project folder, or any step on files you name with `moseq2 standalone`.
    """


@cli.group("standalone", context_settings={"help_option_names": ["-h", "--help"]})
def standalone():
    """Run a step on the files you name, outside any project: for scripts and workflow managers."""


def _build() -> None:
    for command in _commands("STEPS") + _commands("RECORDS"):
        cli.add_command(command)
    for command in _commands("HIDDEN"):
        command.hidden = True
        cli.add_command(command)
    for command in _commands("STANDALONE"):
        standalone.add_command(command)


_build()

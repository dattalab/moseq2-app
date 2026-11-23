"""The ``moseq2`` command: its help, and how a run ends.

``moseq2 --help`` lists the steps run in a project, then each standalone
command, then the records commands, under plain headings. An expected error
prints its message and the fix and exits 1, with no traceback unless ``--debug``
asks; anything else is a bug: its traceback and a line asking for a report, exit
3. Every command a message names exists.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import click
from click.testing import CliRunner

from moseq2_app.cli import BUG_LINE, cli
from extract_projects import small_project

PACKAGES = Path(__file__).resolve().parents[2]


def _run(args, cwd=None):
    """``moseq2 <args>`` run in ``cwd``, as typed."""
    before = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    try:
        return CliRunner().invoke(cli, [str(a) for a in args], prog_name="moseq2")
    finally:
        os.chdir(before)


def test_help_lists_steps_standalone_and_records_under_their_headings():
    result = _run(["--help"])
    assert result.exit_code == 0
    text = result.output
    steps = text.index("Steps, in your project:")
    standalone = text.index("Standalone: any step on files you name, for scripts:")
    records = text.index("Records:")
    assert steps < standalone < records
    assert re.search(r"^  extract\s+Extract", text[steps:standalone], re.MULTILINE)
    for name in ("extract", "convert-raw-to-avi", "copy-slice"):
        assert re.search(rf"^  standalone {name}\s", text[standalone:records], re.MULTILINE)
    assert "print-record" in text[records:]
    assert "run-job" not in text


def test_an_expected_error_prints_its_message_and_exits_1_without_a_traceback(tmp_path):
    files = tmp_path / "recordings.txt"
    files.write_text("# the recordings\nsession_1/depth.avi\n")

    result = _run(["standalone", "extract", "--files-from", files])

    assert result.exit_code == 1
    assert result.stderr == (f"Error: {files} names session_1/depth.avi on line 2, which does not exist.\n"
                             "To fix it: correct or delete that line.\n")
    assert "Traceback" not in result.output


def test_a_bug_prints_the_traceback_and_the_bug_line_and_exits_3(tmp_path, raw_sessions, monkeypatch):
    project = small_project(tmp_path / "project", raw_sessions)

    def broken(*args, **kwargs):
        print("printed before the bug")
        raise KeyError("a bug")

    monkeypatch.setattr("moseq2_extract.project.extract", broken)
    result = _run(["extract"], cwd=project.folder)

    assert result.exit_code == 3
    assert "Traceback (most recent call last)" in result.stderr
    assert "KeyError: 'a bug'" in result.stderr
    assert "printed before the bug" in result.stderr
    assert result.stderr.rstrip().endswith(BUG_LINE)
    assert BUG_LINE == ("This looks like a bug in MoSeq2; please report it at "
                        "https://github.com/dattalab/moseq2-app/issues with the traceback above.")


def test_debug_shows_the_traceback_of_an_expected_error(tmp_path):
    files = tmp_path / "recordings.txt"
    files.write_text("missing/depth.avi\n")

    result = _run(["standalone", "extract", "--files-from", files, "--debug"])

    assert result.exit_code == 1
    assert "Traceback (most recent call last)" in result.stderr
    assert "Error: " in result.stderr


def test_python_m_moseq2_app_is_the_same_command(tmp_path):
    script = Path(sys.executable).parent / "moseq2"
    module = subprocess.run([sys.executable, "-m", "moseq2_app", "--help"], capture_output=True, text=True,
                            cwd=tmp_path)
    console = subprocess.run([str(script), "--help"], capture_output=True, text=True, cwd=tmp_path)

    assert module.returncode == console.returncode == 0
    assert module.stdout == console.stdout
    assert module.stdout.startswith("Usage: moseq2 ")


def test_outside_a_project_a_step_says_how_to_start_one(tmp_path):
    result = _run(["extract"], cwd=tmp_path)

    assert result.exit_code == 1
    assert result.stderr == (f"Error: `moseq2 extract` runs in a project folder, and {tmp_path} has no config.yaml.\n"
                             "To fix it: run it from your project folder, or start a project here with "
                             "`moseq2 setup`.\n")


# ------------------------------------------------------------------------- every command a message names

#: The packages whose messages are read: the six, at their source in this checkout.
SOURCES = [PACKAGES / "moseq2_app", *(PACKAGES / "packages").glob("moseq2-*/moseq2_*")]

#: Step commands later pull requests of this stage add, which a message may already name (R1): Setup's,
#: named by the approved message for a folder holding no project (W7).
PLANNED = {"moseq2 setup"}

#: Words after a package's name in prose ("moseq2-extract writes milliseconds"), not commands.
PROSE = {"writes", "reads", "command", "commands", "packages"}

_NAMED = re.compile(r"\bmoseq2(?:-(?P<package>extract|pca|model|viz))? (?P<word>[a-z][a-z0-9-]*)"
                    r"(?: (?P<sub>[a-z][a-z0-9-]*))?")


def _strings(path: Path):
    """Every string literal in ``path`` but the docstrings, with its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def _installed():
    """Every command of the installed MoSeq2 scripts, as typed: ``moseq2 extract``, ``moseq2 standalone extract``."""
    found = set()
    for ep in entry_points(group="console_scripts"):
        if ep.name == "moseq2" or ep.name.startswith("moseq2-"):
            group = ep.load()
            for name, command in group.commands.items():
                found.add(f"{ep.name} {name}")
                if isinstance(command, click.Group):
                    found |= {f"{ep.name} {name} {sub}" for sub in command.commands}
    return found


def test_every_command_a_message_names_exists():
    installed = _installed() | PLANNED
    missing = []
    for source in SOURCES:
        for path in sorted(source.rglob("*.py")):
            for line, text in _strings(path):
                for match in _NAMED.finditer(text):
                    script = f"moseq2-{match['package']}" if match["package"] else "moseq2"
                    named = f"{script} {match['word']}"
                    if named == "moseq2 standalone" and match["sub"]:
                        named = f"{named} {match['sub']}"
                    if named not in installed and match["word"] not in PROSE:
                        missing.append(f"{path.relative_to(PACKAGES)}:{line}: {named}")
    assert missing == []

"""Projects and notebooks from MoSeq2 1.5.0 stop on 1.6.0 with a message saying what to do.

1.6.0 does not read or convert a project made by 1.5.0 or earlier. Such a
project shows itself by a ``config.yaml`` without sections, or by 1.5.0's
``progress.yaml`` or ``moseq2-index.yaml`` with no ``moseq2-project.yaml``
beside it. Every command of the four command-line programs, the setup commands
(``generate-config``, ``download-flip-file`` and the notebook functions behind
them) and the notebooks' project cell (``moseq2_app.project.start``) stop on one
before doing anything, naming the file that showed it and the two ways forward,
and leave every file in it as it was.

A 1.5 notebook run against 1.6.0 stops at its first project cell: each name a
1.5.0 notebook imports that 1.6.0 removed stops the import with
``OutdatedNotebookError``, which says to download the 1.6 notebooks.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import write_yaml

NOTEBOOK_CELLS = Path(__file__).parent / "notebooks_1_5_0"

INSTALL_1_5 = 'python -m pip install "moseq2-app==1.5.0"'
NOTEBOOK_DOWNLOAD = "https://github.com/dattalab/moseq2-app/tree/release/notebooks"

#: Every name the four 1.5.0 notebooks import that 1.6.0 removed, as found by importing each
#: ``from moseq2_... import ...`` line of the notebooks at the v1.5.0 tag against this tree.
REMOVED_NOTEBOOK_NAMES = [
    ("moseq2_app.gui.progress", "restore_progress_vars"),
    ("moseq2_app.gui.progress", "check_progress"),
    ("moseq2_app.gui.progress", "update_progress"),
    ("moseq2_app.gui.progress", "progress_path_sanity_check"),
    ("moseq2_app.util", "read_yaml"),
    ("moseq2_app.util", "setup_model_folders"),
    ("moseq2_app.util", "update_model_paths"),
    ("moseq2_app.util", "camera_for_pca"),
    ("moseq2_app.flip.train", "CleanParameters"),
]

#: Each stored first cell, and the removed name it imports first.
FIRST_CELLS = {
    "extract_modeling.py": "moseq2_app.gui.progress.check_progress",
    "analysis_visualization.py": "moseq2_app.gui.progress.update_progress",
    "apply_pre_trained_model.py": "moseq2_app.gui.progress.check_progress",
    "flip_classifier_training.py": "moseq2_app.flip.train.CleanParameters",
}

#: How each kind of 1.5.0 project shows itself: the file, and the words the message gives for it.
OLD_KINDS = {
    "flat config": ("config.yaml", "its settings have no sections"),
    "progress file": ("progress.yaml", "it is 1.5.0's progress file"),
    "index file": ("moseq2-index.yaml", "it is 1.5.0's index file"),
}

FLAT_CONFIG = {"bg_roi_depth_range": [650, 750], "chunk_size": 1000, "gaussfilter_space": [1.5, 1],
               "kappa": 1000000.0, "camera_type": "k2"}


def old_project(folder: Path, kind: str) -> Path:
    """A 1.5.0 project in ``folder`` that shows itself as ``kind`` (a key of ``OLD_KINDS``)."""
    folder.mkdir(parents=True, exist_ok=True)
    session = folder / "session_1"
    (session / "proc").mkdir(parents=True)
    (session / "depth.dat").write_bytes(b"\x00" * 64)
    (session / "proc" / "results_00.yaml").write_text("uuid: 0000\n")
    if kind == "flat config":
        write_yaml(FLAT_CONFIG, str(folder / "config.yaml"))
        write_yaml({"base_dir": str(folder), "config_file": str(folder / "config.yaml")},
                   str(folder / "progress.yaml"))
        write_yaml({"files": [], "pca_path": ""}, str(folder / "moseq2-index.yaml"))
    else:
        write_yaml({"camera_type": "auto", "extract": {}}, str(folder / "config.yaml"))
        if kind == "progress file":
            write_yaml({"base_dir": str(folder)}, str(folder / "progress.yaml"))
        else:
            write_yaml({"files": [], "pca_path": ""}, str(folder / "moseq2-index.yaml"))
    return folder


def snapshot(folder: Path) -> dict:
    """Every file and folder under ``folder``, files with their bytes."""
    return {str(path.relative_to(folder)): (path.read_bytes() if path.is_file() else None)
            for path in sorted(folder.rglob("*"))}


@contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def _groups():
    from moseq2_extract.cli import cli as extract
    from moseq2_model.cli import cli as model
    from moseq2_pca.cli import cli as pca
    from moseq2_viz.cli import cli as viz

    return {"moseq2-extract": extract, "moseq2-pca": pca, "moseq2-model": model, "moseq2-viz": viz}


def _commands():
    return [(program, name) for program, group in _groups().items() for name in sorted(group.commands)]


def _value(param: click.Parameter) -> list[str]:
    """A value ``param`` accepts: a file or folder that exists in the project, a choice, or a number."""
    kind = param.type
    if isinstance(kind, click.Path):
        value = "." if kind.file_okay is False else "config.yaml"
    elif isinstance(kind, click.Choice):
        value = str(kind.choices[0])
    elif isinstance(kind, (click.types.IntParamType, click.types.FloatParamType)):
        value = "1"
    else:
        value = "x"
    return [value] * (param.nargs if param.nargs > 0 else 1)


#: Options given beyond the required ones. Without ``--value``, click rejects add-group's default for it
#: (``Value must be an iterable``) while parsing, before the command runs.
EXTRA_ARGUMENTS = {("moseq2-viz", "add-group"): ["--value", "x"]}


def required_arguments(command: click.Command) -> list[str]:
    """The arguments and options ``command`` cannot run without, each given a value it accepts."""
    args = []
    for param in command.params:
        if isinstance(param, click.Argument) and param.required:
            args += _value(param)
        elif isinstance(param, click.Option) and param.required:
            args += [param.opts[0], *_value(param)]
    return args


@pytest.mark.parametrize("kind", list(OLD_KINDS))
@pytest.mark.parametrize("program,command", _commands())
def test_every_command_stops_on_a_1_5_project_and_writes_nothing(tmp_path, program, command, kind):
    """Run in a 1.5.0 project's folder, every command stops before doing anything, saying what to do."""
    folder = old_project(tmp_path / "project", kind)
    before = snapshot(folder)
    group = _groups()[program]

    with working_in(folder):
        result = CliRunner().invoke(group, [command, *required_arguments(group.commands[command]),
                                            *EXTRA_ARGUMENTS.get((program, command), [])])

    file, reason = OLD_KINDS[kind]
    assert result.exit_code != 0, result.output
    assert INSTALL_1_5 in result.output, (result.output, result.exception)
    assert f"{file} shows that" in result.output and reason in result.output, result.output
    assert snapshot(folder) == before


@pytest.mark.parametrize("option", ["--config-file", "--project-file"])
def test_a_command_given_a_1_5_project_from_elsewhere_stops(tmp_path, option):
    """A 1.5.0 project named by --config-file or --project-file stops the command, run from any folder."""
    from moseq2_model.cli import cli as model
    from moseq2_pca.cli import cli as pca

    folder = old_project(tmp_path / "project", "progress file")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    before = snapshot(folder)

    with working_in(elsewhere):
        if option == "--config-file":
            result = CliRunner().invoke(pca, ["train-pca", "--config-file", str(folder / "config.yaml")])
        else:
            result = CliRunner().invoke(model, ["apply-model", str(folder / "config.yaml"),
                                                str(folder / "config.yaml"), "out.p", "--project-file",
                                                str(folder / "moseq2-project.yaml")])

    assert result.exit_code != 0
    assert INSTALL_1_5 in result.output, result.output
    assert "progress.yaml shows that" in result.output
    assert snapshot(folder) == before
    assert list(elsewhere.iterdir()) == []


def test_help_still_answers_in_a_1_5_project(tmp_path):
    """--help and --version read nothing from the project, so they answer in one."""
    from moseq2_extract.cli import cli

    folder = old_project(tmp_path / "project", "flat config")
    with working_in(folder):
        assert CliRunner().invoke(cli, ["extract", "--help"]).exit_code == 0
        assert CliRunner().invoke(cli, ["--version"]).exit_code == 0


@pytest.mark.parametrize("kind", list(OLD_KINDS))
def test_setup_commands_stop_on_a_1_5_project(tmp_path, monkeypatch, kind):
    """generate-config and download-flip-file, and the notebook functions behind them, stop and write nothing."""
    from moseq2_common.project import OldProjectError
    from moseq2_extract import setup
    from moseq2_extract.gui import download_flip_command, generate_config_command

    folder = old_project(tmp_path / "project", kind)
    before = snapshot(folder)
    config = str(folder / "config.yaml")
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    file, reason = OLD_KINDS[kind]
    for call in (lambda: setup.generate_config(config, "auto", overwrite=True),
                 lambda: generate_config_command(config, camera_type="auto"),
                 lambda: setup.download_flip_file(config, str(folder), 0),
                 lambda: download_flip_command(str(folder), config, 0)):
        with pytest.raises(OldProjectError) as error:
            call()
        assert INSTALL_1_5 in str(error.value) and f"{file} shows that" in str(error.value)
        assert reason in str(error.value)
    assert snapshot(folder) == before


@pytest.mark.parametrize("kind", list(OLD_KINDS))
def test_the_notebooks_project_cell_stops_on_a_1_5_project(tmp_path, kind):
    """``moseq2_app.project.start``, every 1.6 notebook's first project cell, stops on a 1.5.0 project."""
    from moseq2_app.project import start
    from moseq2_common.project import OldProjectError

    folder = old_project(tmp_path / "project", kind)
    before = snapshot(folder)

    with pytest.raises(OldProjectError, match="1.6.0 does not read or convert"):
        start(str(folder))
    assert snapshot(folder) == before


def test_a_1_6_project_is_not_old(tmp_path):
    """A sectioned config.yaml, or 1.5.0's files beside a moseq2-project.yaml, is a 1.6.0 project."""
    from moseq2_common.project import check_not_old

    check_not_old(tmp_path)
    write_yaml({"camera_type": "auto", "extract": {"chunk_size": 1000}, "groups": {}}, str(tmp_path / "config.yaml"))
    check_not_old(tmp_path)
    for name in ("progress.yaml", "moseq2-index.yaml"):
        (tmp_path / name).write_text("{}\n")
    (tmp_path / "moseq2-project.yaml").write_text("sessions: {}\n")
    check_not_old(tmp_path)


@pytest.mark.parametrize("cell,name", sorted(FIRST_CELLS.items()))
def test_1_5_notebooks_stop_at_their_first_cell(tmp_path, cell, name):
    """Each 1.5.0 notebook's first project cell, run against 1.6.0, stops on its first removed import."""
    from moseq2_common.project import OutdatedNotebookError

    source = (NOTEBOOK_CELLS / cell).read_text()
    old_project(tmp_path / "project", "flat config")

    with working_in(tmp_path / "project"), pytest.raises(OutdatedNotebookError) as error:
        exec(compile(source, cell, "exec"), {"__name__": "__notebook__"})

    message = str(error.value)
    assert f"`{name}`" in message, message
    assert NOTEBOOK_DOWNLOAD in message and INSTALL_1_5 in message


@pytest.mark.parametrize("module,name", REMOVED_NOTEBOOK_NAMES)
def test_every_removed_notebook_name_stops_with_the_download_message(module, name):
    """Each name a 1.5.0 notebook imports that 1.6.0 removed stops its import, naming it."""
    from moseq2_common.project import OutdatedNotebookError

    with pytest.raises(OutdatedNotebookError, match=f"`{module}.{name}`"):
        exec(f"from {module} import {name}", {})
    with pytest.raises(AttributeError):
        getattr(importlib.import_module(module), "no_such_name")

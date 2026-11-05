"""The command lines share one copy of their option helpers, in moseq2-common.

Reading a command's settings from its project's ``config.yaml`` and turning a
bad file into a command-line error are the same few lines in every package's
``cli.py``. They are kept once in ``moseq2_common.options``
(``command_settings``), so a fix lands once.

Each ``cli.py`` is read from its installed source with ``ast``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import click
import pytest

from moseq2_common import options

HELPERS = ["command_settings"]
COPIED_NAMES = ["_read_config_file", "read_config_option", *HELPERS]
PACKAGES = ["moseq2_extract", "moseq2_pca", "moseq2_model", "moseq2_viz"]


@pytest.mark.parametrize("name", HELPERS)
def test_helper_lives_in_moseq2_common_options(name):
    assert callable(getattr(options, name, None))


@pytest.mark.parametrize("pkg", PACKAGES)
def test_cli_defines_no_copy_of_a_shared_helper(pkg):
    path = Path(importlib.import_module(pkg).__file__).parent / "cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = [
        f"{pkg}/cli.py:{node.lineno} {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COPIED_NAMES
    ]
    assert found == []


def test_command_settings_outside_a_project_are_empty_and_say_so(tmp_path, monkeypatch, capsys):
    from moseq2_pca.options import TrainPCAOptions

    monkeypatch.chdir(tmp_path)
    assert options.command_settings(None, TrainPCAOptions) == {}
    assert "runs outside a project" in capsys.readouterr().out


def test_command_settings_stop_the_command_on_a_missing_file(tmp_path):
    from moseq2_pca.options import TrainPCAOptions

    missing = tmp_path / "config.yaml"
    with pytest.raises(click.ClickException, match=str(missing)):
        options.command_settings(missing, TrainPCAOptions)

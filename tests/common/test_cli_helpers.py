"""The command lines share one copy of their option helpers, in moseq2-common.

Reading ``--config-file`` and turning a bad file into a command-line error,
starting a command from click's type-converted defaults, and declaring a flag
with no default are the same few lines in every package's ``cli.py``. Each is
kept once in ``moseq2_common.options``, so a fix (or the removal of a shim)
lands once.

Each ``cli.py`` is read from its installed source with ``ast``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import click
import pytest

from moseq2_common import options

HELPERS = ["read_config_option", "shim_click_defaults", "shim_flags_without_default"]
COPIED_NAMES = ["_read_config_file", *HELPERS]
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


def test_read_config_option_without_a_file_is_none():
    assert options.read_config_option(None) is None


def test_read_config_option_stops_the_command_on_a_missing_file(tmp_path):
    missing = tmp_path / "config.yaml"
    with pytest.raises(click.ClickException, match=str(missing)):
        options.read_config_option(missing)

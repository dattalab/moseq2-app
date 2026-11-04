"""Every option a step declares is read by the code the step runs.

The options classes are the only list of a step's settings, and the code that
uses them reads a dict or an options object by name. An option no code reads
is invisible: it is in ``--help``, ``config.yaml`` and the arena tool's saved
settings, and changing it changes nothing. 1.5.0 declared seven, among them
``--tracking-model-mask-threshold``, whose -16 never reached extraction
(``extract_chunk`` used its own ``mask_threshold`` of -20), and ``apply-pca
--detrend-window`` (survey D13, L9).

A field counts as read when, in the source of the six packages outside the
``options.py`` modules, its name is read as ``x.name``, ``x["name"]`` or
``x.get("name")``, or is a parameter of a function (the steps hand their
options on as dicts that reach functions such as ``extract_chunk`` as
``**config_data``).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path

from moseq2_common.options import Options

OPTIONS_MODULES = [
    "moseq2_common.slurm",
    "moseq2_extract.options",
    "moseq2_pca.options",
    "moseq2_model.options",
    "moseq2_viz.options",
]


def _package_dirs():
    names = ["moseq2_app", "moseq2_common", "moseq2_extract", "moseq2_pca", "moseq2_model", "moseq2_viz"]
    return [Path(importlib.import_module(name).__file__).parent for name in names]


def _names_read():
    """Every name read as an attribute, a string key or a function parameter."""
    read = set()
    for folder in _package_dirs():
        for path in sorted(folder.rglob("*.py")):
            if path.name == "options.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    read.add(node.attr)
                elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        read.add(node.slice.value)
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "get" and node.args
                      and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    read.add(node.args[0].value)
                elif isinstance(node, ast.arg):
                    read.add(node.arg)
    return read


def _declared():
    seen = {}
    for module_name in OPTIONS_MODULES:
        module = importlib.import_module(module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Options) and cls is not Options and cls.__module__ == module_name:
                for field in dataclasses.fields(cls):
                    seen.setdefault(field.name, f"{cls.__name__} ({module_name})")
    return seen


def test_every_declared_option_is_read_by_a_step():
    read = _names_read()
    unread = sorted(
        f"{name} (declared in {where})"
        for name, where in _declared().items()
        if name not in read
    )
    assert not unread, (
        "These options are declared but no code reads them, so setting them changes nothing: "
        + "; ".join(unread)
        + ". Read each where its step needs it, or delete it from its options class."
    )


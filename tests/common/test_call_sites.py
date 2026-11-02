"""Every package reads and writes YAML through moseq2-common.

A private copy of the YAML helpers, or a ``YAML()`` built at a call site, is
a second format and a second place for the numpy and truncation fixes to be
missing from. And a ``write_yaml`` handed a file the caller already opened
with ``"w"`` has emptied that file before the writer sees the data, so it
cannot leave the file alone when the data cannot be written; it has to be
given the path.

Read from each package's installed source with ``ast``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

PACKAGES = ["moseq2_extract", "moseq2_pca", "moseq2_model", "moseq2_viz", "moseq2_app"]


def sources(pkg):
    root = Path(importlib.import_module(pkg).__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        yield path.relative_to(root.parent).as_posix(), tree


def call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_keeps_no_yaml_module_of_its_own(pkg):
    assert importlib.util.find_spec(f"{pkg}._yaml") is None


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_builds_no_yaml_parser_of_its_own(pkg):
    found = [
        f"{path}:{node.lineno}"
        for path, tree in sources(pkg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and call_name(node) == "YAML"
    ]
    assert found == []


@pytest.mark.parametrize("pkg", PACKAGES)
def test_write_yaml_is_given_a_path_not_an_open_file(pkg):
    found = []
    for path, tree in sources(pkg):
        for block in ast.walk(tree):
            if not isinstance(block, ast.With):
                continue
            handles = {
                item.optional_vars.id
                for item in block.items
                if isinstance(item.context_expr, ast.Call)
                and call_name(item.context_expr) == "open"
                and isinstance(item.optional_vars, ast.Name)
            }
            if not handles:
                continue
            for node in ast.walk(block):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node) or ""
                args = list(node.args) + [kw.value for kw in node.keywords]
                if name.endswith("write_yaml") and any(
                    isinstance(a, ast.Name) and a.id in handles for a in args
                ):
                    found.append(f"{path}:{node.lineno}")
    assert found == []

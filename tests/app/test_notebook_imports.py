"""Every name the notebooks import exists in the installed packages.

A notebook is run cell by cell long after the code under it moved, so a
function deleted or renamed in a package shows up as an ImportError in a
user's notebook rather than in this suite. Read from the notebooks a user is
given, so the check follows them as they change.
"""

from __future__ import annotations

import ast
import importlib
import json
import textwrap

import pytest

from cells import NOTEBOOKS

IMPORT_LINE = ("import ", "from ")


def notebook_imports(notebook):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for number, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
        for line in "".join(cell["source"]).splitlines():
            if not line.strip().startswith(IMPORT_LINE):
                continue
            for node in ast.parse(textwrap.dedent(line)).body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        yield number + 1, alias.name, None
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        yield number + 1, node.module, alias.name


@pytest.mark.parametrize("notebook", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_every_name_a_notebook_imports_exists(notebook):
    missing = []
    for cell, module, name in notebook_imports(notebook):
        try:
            imported = importlib.import_module(module)
        except ImportError as err:
            missing.append(f"code cell {cell}: import {module} ({err})")
            continue
        if name is not None and not hasattr(imported, name):
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(f"code cell {cell}: from {module} import {name}")
    assert missing == []

"""The release check refuses a tree whose packages disagree about moseq2-common.

``.github/scripts/check_release.py`` runs before anything is built or
uploaded. Each of the five packages that depend on moseq2-common must require
it at exactly the tag's version, because the same run uploads it: a pin naming
anything else names a release this run does not make, and users installing the
new release would get a resolver error instead of MoSeq2.

The script needs ``tomllib`` (Python 3.11) and lives in the repository, not the
sdist, so the module skips on 3.10 and in an unpacked sdist.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "check_release.py"
CORE = ("moseq2-extract", "moseq2-pca", "moseq2-model", "moseq2-viz")

pytestmark = [
    pytest.mark.skipif(
        not (REPO / ".github").is_dir(), reason="the release scripts are not part of the sdist"
    ),
    pytest.mark.skipif(sys.version_info < (3, 11), reason="check_release.py needs tomllib"),
]


def _pyproject(name, dependencies):
    deps = "".join(f'    "{d}",\n' for d in dependencies)
    return f'[project]\nname = "{name}"\ndependencies = [\n{deps}]\n'


def _init(directory, module):
    (directory / module).mkdir(parents=True)
    (directory / module / "__init__.py").write_text('__version__ = "1.4.5"\n')


def make_tree(root, *, common_pins=None):
    """A minimal repository the release check passes at v1.4.5.

    ``common_pins`` maps a core package to the moseq2-common requirement it
    declares, or to ``None`` to leave it out.
    """
    common_pins = common_pins or {}
    siblings = ("moseq2-common",) + CORE
    (root / "pyproject.toml").write_text(
        _pyproject("moseq2-app", [f"{name}==1.4.5" for name in siblings])
    )
    _init(root, "moseq2_app")
    for name in siblings:
        directory = root / "packages" / name
        _init(directory, name.replace("-", "_"))
        if name == "moseq2-common":
            deps = ["numpy>=2.2.6"]
        else:
            pin = common_pins.get(name, "moseq2-common==1.4.5")
            deps = [] if pin is None else [pin]
        (directory / "pyproject.toml").write_text(_pyproject(name, deps))
    (root / "notebooks").mkdir()
    (root / "notebooks" / "a.ipynb").write_text(
        json.dumps(
            {"metadata": {"moseq2": {"version": "1.4"}}, "cells": [], "nbformat": 4,
             "nbformat_minor": 5}
        )
    )


def run(root, tag):
    return subprocess.run(
        [sys.executable, str(SCRIPT), tag], cwd=root, capture_output=True, text=True, timeout=120
    )


def test_a_consistent_tree_passes(tmp_path):
    make_tree(tmp_path)
    proc = run(tmp_path, "v1.4.5")
    assert proc.returncode == 0, proc.stderr


def test_a_package_pinning_another_moseq2_common_is_refused(tmp_path):
    make_tree(tmp_path, common_pins={"moseq2-extract": "moseq2-common==1.4.4"})
    proc = run(tmp_path, "v1.4.5")
    assert proc.returncode != 0
    assert "moseq2-extract requires moseq2-common==1.4.4, not moseq2-common==1.4.5" in proc.stderr


def test_a_package_without_moseq2_common_is_refused(tmp_path):
    make_tree(tmp_path, common_pins={"moseq2-pca": None})
    proc = run(tmp_path, "v1.4.5")
    assert proc.returncode != 0
    assert "moseq2-pca does not require moseq2-common" in proc.stderr


def test_the_repository_passes_at_its_own_version():
    init = (REPO / "moseq2_app" / "__init__.py").read_text()
    app_version = re.search(r'^__version__ = "([^"]+)"', init, re.M).group(1)
    proc = run(REPO, f"v{app_version}")
    assert proc.returncode == 0, proc.stderr

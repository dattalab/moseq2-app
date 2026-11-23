"""The step GUIs' scripts and styles ship in moseq2-app's wheel."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_the_wheel_ships_the_gui_files(tmp_path):
    pytest.importorskip("pdm.backend")
    built = subprocess.run([sys.executable, "-c", f"from pdm.backend import build_wheel; "
                                                  f"print(build_wheel({str(tmp_path)!r}))"],
                           cwd=ROOT, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    wheel = tmp_path / built.stdout.strip().splitlines()[-1]

    shipped = set(zipfile.ZipFile(wheel).namelist())
    static = sorted(p.name for p in (ROOT / "moseq2_app" / "notebook" / "static").iterdir())
    assert static
    for name in static:
        assert f"moseq2_app/notebook/static/{name}" in shipped

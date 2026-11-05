"""Cell 1: name the finished project.

    base_dir = './'
    project = start(base_dir)

The analysis notebook starts where the extraction and modeling notebook
stopped, and this cell is the handover: the project folder, whose
``moseq2-project.yaml`` says which sessions it holds and which model is
current. Every later cell takes the folder and reads that file when it
needs a path, so nothing is carried between cells but the folder.
"""

from __future__ import annotations

from pathlib import Path

from moseq2_app.project import start


def test_the_cell_names_the_project_folder(analysis_project, analysis_started):
    """1. The folder every later cell is handed."""
    assert Path(analysis_started.folder) == Path(analysis_project)
    assert Path(analysis_started.bookkeeping_file) == Path(analysis_project) / "moseq2-project.yaml"


def test_the_cell_says_what_the_analysis_will_read(analysis_project, capsys):
    """2. The sessions and the model the later cells will analyse, before any of them runs."""
    start(str(analysis_project))
    out = capsys.readouterr().out
    assert "Sessions: 20 aggregated" in out
    assert "current models/model.p" in out


def test_the_cell_writes_nothing(analysis_project):
    """3. Opening the notebook changes no file in the project."""
    project = Path(analysis_project)
    before = {p: p.read_bytes() for p in project.glob("*.yaml")}
    start(str(project))
    assert {p: p.read_bytes() for p in project.glob("*.yaml")} == before

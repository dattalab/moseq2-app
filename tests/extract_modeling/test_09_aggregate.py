"""Cell 9: list the extracted recordings as the project's sessions.

    aggregate_extract_results_command(str(project.folder))

The hinge of the notebook. Extraction has already written each recording's
files into its own folder in ``aggregate_results/``, named from the
recording's path in the project; this cell copies nothing and lists each
extracted recording as one of the project's sessions in
``moseq2-project.yaml``.

That file is the project's table of contents from here on: one session per
recording, keyed by the recording's path in the project, with its uuid and its
folder in ``aggregate_results/``. The later steps read the sessions as an index
in memory (``moseq2_common.bookkeeping.session_index``), each with the
acquisition metadata and its group from ``config.yaml``, ``default`` until
cell 10 sets them.
"""

from __future__ import annotations

from pathlib import Path

import h5py

from moseq2_common.bookkeeping import session_index
from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml

from test_03_config import plain
from test_05_arena_mask import RECORDINGS
from test_06_extract import OUTPUTS, proc


def entries(where):
    """The project's sessions, as the later steps read them, keyed by session name.

    ``where`` is the project folder or its ``moseq2-project.yaml``.
    """
    root = Path(where) if Path(where).is_dir() else Path(where).parent
    listed = plain(session_index(Project.at(root)))["files"]
    return {entry["metadata"]["SessionName"]: entry for entry in listed}


def test_every_session_has_one_folder_in_aggregate_results(aggregated, project):
    """1. The cell's purpose: one folder holding the whole experiment, one subfolder per session,
    named from the recording's folder, holding what extraction wrote and no copy of it."""
    folders = sorted(p.name for p in (project / "aggregate_results").iterdir())
    assert folders == RECORDINGS
    for name in RECORDINGS:
        assert sorted(p.name for p in proc(project, name).iterdir()) == sorted(OUTPUTS), name


def test_the_sessions_carry_each_extractions_uuid(aggregated, project):
    """2. What a model's results are matched back to.

    The uuid is written by the extraction and travels into the bookkeeping
    file, and the PC scores and the model key each session by it.
    """
    listed = entries(aggregated)
    assert sorted(listed) == RECORDINGS

    for name, entry in listed.items():
        with h5py.File(proc(project, name) / "results.h5") as f:
            assert entry["uuid"] == f["metadata/uuid"][()].decode()
    assert len({entry["uuid"] for entry in listed.values()}) == len(RECORDINGS)


def test_each_entry_points_at_its_session_folder(aggregated, project):
    """3. The sessions are the files extraction wrote, in their folders."""
    for name, entry in entries(aggregated).items():
        assert [Path(p) for p in entry["path"]] == [proc(project, name) / "results.h5"]


def test_aggregation_lists_the_sessions_and_nothing_else(bookkeeping_as_aggregated, project):
    """4. What the cell writes: each recording as a session, keyed by its path in the project.

    No group is stored here (groups are config.yaml's), and no later step has
    recorded an output yet.
    """
    stored = read_yaml(str(bookkeeping_as_aggregated))

    assert set(stored["sessions"]) == {f"{name}/depth.avi" for name in RECORDINGS}
    for key, session in stored["sessions"].items():
        assert session["recording"] == key
        assert session["folder"] == f"aggregate_results/{key.split('/')[0]}"
    assert stored["pca"] == {}
    assert stored["models"]["fitted"] == []
    assert "group" not in str(stored)


def test_a_session_no_group_names_is_in_the_default_group(bookkeeping_as_aggregated, tmp_path):
    """5. The state cell 10 exists to change: with no groups in config.yaml, every session is ``default``."""
    import shutil

    copy = tmp_path / "project"
    copy.mkdir()
    shutil.copy2(bookkeeping_as_aggregated, copy / "moseq2-project.yaml")
    (copy / "config.yaml").write_text("camera_type: auto\n")

    assert {entry["group"] for entry in session_index(Project.at(copy))["files"]} == {"default"}


def test_the_bookkeeping_file_is_written_in_the_project(aggregated, project):
    """6. Where the rest of the notebook finds it."""
    assert aggregated == project / "moseq2-project.yaml"

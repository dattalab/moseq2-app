"""Cell 9: gather the extractions into one directory and list the sessions.

    recording_format = '{start_time}_{session_name}_{subject_name}'
    aggregate_results_dirname = 'aggregate_results'

    aggregate_extract_results_command(str(project.folder), recording_format, aggregate_results_dirname)

The hinge of the notebook. Up to here every output has lived beside the
recording that produced it, named ``results_00`` in a ``proc`` directory and
told apart only by which folder it sits in. Modeling reads one directory, so
the cell copies everything into ``aggregate_results/`` under a name built from
the acquisition metadata, and lists each recording as one of the project's
sessions in ``moseq2-project.yaml``.

That file is the project's table of contents from here on: one session per
recording, keyed by the recording's path in the project, with its uuid and its
file in ``aggregate_results/``. The later steps read the sessions as an index
in memory (``moseq2_common.bookkeeping.session_index``), each with the
acquisition metadata and its group from ``config.yaml``, ``default`` until
cell 10 sets them.

The copies are byte-identical, so the originals beside each recording stay
usable and nothing is rewritten in the move.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py

from moseq2_common.bookkeeping import session_index
from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml

from test_03_config import plain
from test_05_arena_mask import RECORDINGS
from test_06_extract import proc

#: What is copied for each recording. The index names only the first two.
COPIED = ("h5", "mp4", "yaml")

#: `{start_time}_{session_name}_{subject_name}`, with the start time taken off
#: the front of the metadata's ISO timestamp.
NAME = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<session>.+)_(?P<subject>[^_]+)_results_00\.(?P<ext>h5|mp4|yaml)"
)


def entries(where):
    """The project's sessions, as the later steps read them, keyed by session name.

    ``where`` is the project folder or its ``moseq2-project.yaml``.
    """
    root = Path(where) if Path(where).is_dir() else Path(where).parent
    listed = plain(session_index(Project.at(root)))["files"]
    return {entry["metadata"]["SessionName"]: entry for entry in listed}


def test_every_recording_is_copied_into_one_directory(aggregated, project):
    """1. The cell's purpose: one directory holding the whole experiment."""
    gathered = sorted(p.name for p in (project / "aggregate_results").iterdir())

    assert len(gathered) == len(RECORDINGS) * len(COPIED)
    parsed = [NAME.fullmatch(name) for name in gathered]
    assert all(parsed), f"names that do not follow the template: {gathered}"
    assert sorted({m["session"] for m in parsed}) == RECORDINGS
    assert sorted({m["ext"] for m in parsed}) == sorted(COPIED)


def test_the_copied_name_is_built_from_the_acquisition_metadata(aggregated, project):
    """2. Why the template is worth having.

    `results_00.h5` says nothing about which mouse or which day. The copied
    name carries the recording's own start time, session and subject, so the
    files are identifiable outside the directory tree that produced them.
    """
    for name, entry in entries(aggregated).items():
        metadata = entry["metadata"]
        copied = NAME.fullmatch(Path(entry["path"][0]).name)

        assert copied["session"] == name == metadata["SessionName"]
        assert copied["subject"] == metadata["SubjectName"]
        assert metadata["StartTime"].startswith(copied["date"])
        assert copied["time"] == metadata["StartTime"][11:19].replace(":", "-")


def test_the_copies_are_identical_to_what_was_extracted(aggregated, project):
    """3. Nothing is rewritten in the move.

    The originals beside each recording stay usable, and a difference between
    the two copies would mean the modeling read something the extraction did
    not produce.
    """
    for name, entry in entries(aggregated).items():
        copied = Path(entry["path"][0])
        assert copied.read_bytes() == (proc(project, name) / "results_00.h5").read_bytes()


def test_the_sessions_carry_each_extractions_uuid(aggregated, project):
    """4. What a model's results are matched back to.

    The uuid is written by the extraction and travels into the bookkeeping
    file, and the PC scores and the model key each session by it.
    """
    listed = entries(aggregated)
    assert sorted(listed) == RECORDINGS

    for name, entry in listed.items():
        with h5py.File(proc(project, name) / "results_00.h5") as f:
            assert entry["uuid"] == f["metadata/uuid"][()].decode()
    assert len({entry["uuid"] for entry in listed.values()}) == len(RECORDINGS)


def test_each_entry_points_at_the_copies_not_the_originals(aggregated, project):
    """5. The sessions are the aggregate directory's files."""
    for entry in entries(aggregated).values():
        paths = [Path(p) for p in entry["path"]]
        assert [p.suffix for p in paths] == [".h5", ".yaml"]
        for path in paths:
            assert path.exists()
            assert path.parent == project / "aggregate_results"


def test_the_movies_are_gathered_but_not_indexed(aggregated, project):
    """6. Current behaviour, captured.

    Each recording's preview movie is copied with the rest, but a session's
    paths name only the h5 and the yaml. Anything that wants the movie finds
    it by rebuilding the name.
    """
    movies = sorted((project / "aggregate_results").glob("*.mp4"))
    assert len(movies) == len(RECORDINGS)

    indexed = {p for entry in entries(aggregated).values() for p in entry["path"]}
    assert not any(str(movie) in indexed for movie in movies)


def test_aggregation_lists_the_sessions_and_nothing_else(bookkeeping_as_aggregated, project):
    """7. What the cell writes: each recording as a session, keyed by its path in the project.

    No group is stored here (groups are config.yaml's), and no later step has
    recorded an output yet.
    """
    stored = read_yaml(str(bookkeeping_as_aggregated))

    assert set(stored["sessions"]) == {f"{name}/depth.avi" for name in RECORDINGS}
    for key, session in stored["sessions"].items():
        assert session["recording"] == key
        assert session["folder"].startswith("aggregate_results/") and session["folder"].endswith(".h5")
    assert stored["pca"] == {}
    assert stored["models"]["fitted"] == []
    assert "group" not in str(stored)


def test_a_session_no_group_names_is_in_the_default_group(bookkeeping_as_aggregated, tmp_path):
    """8. The state cell 10 exists to change: with no groups in config.yaml, every session is ``default``."""
    import shutil

    copy = tmp_path / "project"
    copy.mkdir()
    shutil.copy2(bookkeeping_as_aggregated, copy / "moseq2-project.yaml")
    (copy / "config.yaml").write_text("camera_type: auto\n")

    assert {entry["group"] for entry in session_index(Project.at(copy))["files"]} == {"default"}


def test_the_bookkeeping_file_is_written_in_the_project(aggregated, project):
    """9. Where the rest of the notebook finds it."""
    assert aggregated == project / "moseq2-project.yaml"

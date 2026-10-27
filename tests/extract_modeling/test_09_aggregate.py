"""Cell 9: gather the extractions into one directory and index them.

    recording_format = '{start_time}_{session_name}_{subject_name}'
    aggregate_results_dirname = 'aggregate_results/'

    train_data_dir = join(progress_paths['base_dir'], aggregate_results_dirname)
    update_progress(progress_filepath, 'train_data_dir', train_data_dir)

    index_filepath = aggregate_extract_results_command(
        progress_paths['base_dir'], recording_format, aggregate_results_dirname)
    progress_paths = update_progress(progress_filepath, 'index_file', index_filepath)

The hinge of the notebook. Up to here every output has lived beside the
recording that produced it, named ``results_00`` in a ``proc`` directory and
told apart only by which folder it sits in. Modeling reads one directory, so
the cell copies everything into ``aggregate_results/`` under a name built from
the acquisition metadata, and writes ``moseq2-index.yaml`` describing what it
gathered.

The index is the project's table of contents from here on: one entry per
recording, carrying a uuid, the acquisition metadata, a group label and the
paths to that recording's files. Every later cell takes the index rather than
the project. The group starts as ``default`` for everything and is what cell
10 exists to change.

The copies are byte-identical, so the originals beside each recording stay
usable and nothing is rewritten in the move.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_extract.util import read_yaml

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


def entries(index):
    """The index's recordings, keyed by session name."""
    listed = plain(read_yaml(str(index)))["files"]
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


def test_the_index_lists_every_recording_by_uuid(aggregated, project):
    """4. What identifies a recording from here on.

    The uuid is written by the extraction and travels into the index, which is
    how a model's results are later matched back to the recording they came
    from. Names and paths change; this does not.
    """
    listed = entries(aggregated)
    assert sorted(listed) == RECORDINGS

    for name, entry in listed.items():
        with h5py.File(proc(project, name) / "results_00.h5") as f:
            assert entry["uuid"] == f["metadata/uuid"][()].decode()
    assert len({entry["uuid"] for entry in listed.values()}) == len(RECORDINGS)


def test_each_entry_points_at_the_copies_not_the_originals(aggregated, project):
    """5. The index is a view of the aggregate directory."""
    for entry in entries(aggregated).values():
        paths = [Path(p) for p in entry["path"]]
        assert [p.suffix for p in paths] == [".h5", ".yaml"]
        for path in paths:
            assert path.exists()
            assert path.parent == project / "aggregate_results"


def test_the_movies_are_gathered_but_not_indexed(aggregated, project):
    """6. Current behaviour, captured.

    Each recording's preview movie is copied with the rest, but the index
    entry names only the h5 and the yaml. Anything that wants the movie finds
    it by rebuilding the name rather than by reading the index.
    """
    movies = sorted((project / "aggregate_results").glob("*.mp4"))
    assert len(movies) == len(RECORDINGS)

    indexed = {p for entry in entries(aggregated).values() for p in entry["path"]}
    assert not any(str(movie) in indexed for movie in movies)


def test_every_recording_starts_in_the_default_group(aggregated):
    """7. The state cell 10 exists to change.

    Nothing has told the project which recordings are which condition yet, so
    every entry carries the same label and a group comparison here would
    compare the experiment against itself.
    """
    assert {entry["group"] for entry in entries(aggregated).values()} == {"default"}


def test_the_index_is_written_beside_the_project_and_recorded(
    aggregated, project, progress_file
):
    """8. Where the rest of the notebook finds it."""
    assert aggregated == project / "moseq2-index.yaml"

    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["index_file"] == str(aggregated)
    assert recorded["train_data_dir"] == str(project / "aggregate_results") + "/"


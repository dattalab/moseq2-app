"""One session per recording, each in its own folder in ``aggregate_results/``.

A session is its recording, keyed by the recording's path in the project
(``saline_example_0/depth.avi``). Inside a project, extraction writes every file
of a session straight into that session's folder in ``aggregate_results/``,
named from its key (``saline_example_0``), and nothing beside or into the
recording; ``aggregate-results`` lists the sessions in ``moseq2-project.yaml``
without copying anything. Output names carry no arena-mask index
(``results.h5``, ``results.mp4``), so re-extracting a recording, with the same
settings or another arena mask, replaces its session: the same key, the same
uuid, one entry and one file.

A recording without a ``metadata.json`` is extracted with an empty subject name
and its folder's name as its session name, held in memory: no ``metadata.json``
is written into the recording's folder, on either front end.

1.5.0 wrote each extraction into ``proc/`` beside its recording, named it
``results_00`` after its arena mask, copied it into ``aggregate_results/`` under
a name built from its metadata, and wrote a default ``metadata.json`` into a
recording folder that had none.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml

#: The two recordings the projects here are built from, one per condition.
RECORDINGS = ("saline_example_0", "amphetamine_example_0")
KEY = "saline_example_0/depth.avi"


@contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def ok(cli, *args, cwd):
    """Invoke ``cli`` with ``args`` from ``cwd``, as a person in that folder types it; it must succeed."""
    with working_in(cwd):
        result = CliRunner().invoke(cli, [str(arg) for arg in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def extract_cli():
    from moseq2_extract.cli import cli

    return cli


def new_project(root, raw_sessions, recordings=RECORDINGS, *, metadata=True):
    """A project folder holding ``recordings`` (each in its folder, as the fixtures have them) and a config.yaml."""
    root.mkdir(parents=True)
    for name in recordings:
        shutil.copytree(raw_sessions / name, root / name,
                        ignore=None if metadata else shutil.ignore_patterns("metadata.json"))
    ok(extract_cli(), "generate-config", "--output-file", root / "config.yaml", cwd=root)
    return root


def listing(folder):
    """Every file under ``folder``, with its size and modification time."""
    return {p.relative_to(folder).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(folder).rglob("*"))}


def sessions(root):
    return read_yaml(str(root / "moseq2-project.yaml"))["sessions"]


def session_h5(root, key=KEY):
    """The extraction file the bookkeeping file names for ``key``."""
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    (h5,) = [f["path"][0] for f in session_index(Project.at(root))["files"] if f["key"] == key]
    return Path(h5)


def h5s_in(folder):
    return sorted(Path(folder).rglob("*.h5"))


# ------------------------------------------------------------ re-extracting


def test_reextracting_a_recording_replaces_its_session(tmp_path, raw_sessions):
    """Extract, aggregate, extract again with other settings, aggregate again: one session, the same uuid,
    and the file the project names is the second extraction (1.5.0 kept the first copy and gave the
    second extraction a new uuid)."""
    root = new_project(tmp_path / "project", raw_sessions, (RECORDINGS[0],))
    recording = root / KEY
    ok(extract_cli(), "extract", recording, "--config-file", root / "config.yaml", "--num-frames", 60, cwd=root)
    ok(extract_cli(), "aggregate-results", cwd=root)
    first = sessions(root)

    ok(extract_cli(), "extract", recording, "--config-file", root / "config.yaml", "--num-frames", 30, cwd=root)
    ok(extract_cli(), "aggregate-results", cwd=root)
    second = sessions(root)

    assert list(second) == [KEY]
    assert second[KEY]["uuid"] == first[KEY]["uuid"]
    h5 = session_h5(root)
    with h5py.File(h5, "r") as f:
        assert f["frames"].shape[0] == 30
        uuid = f["metadata/uuid"][()]
    assert (uuid.decode() if isinstance(uuid, bytes) else uuid) == first[KEY]["uuid"]
    assert h5s_in(root / "aggregate_results") == [h5]


def test_reextracting_with_another_arena_mask_replaces_the_session(tmp_path, raw_sessions):
    """The arena mask is a setting of the session, not part of its outputs' names: extracting with the
    second mask the arena search finds replaces the session's file (1.5.0 wrote ``results_01`` beside
    ``results_00``, and aggregation copied both)."""
    root = new_project(tmp_path / "project", raw_sessions, (RECORDINGS[0],))
    recording = root / KEY
    common = ("--config-file", root / "config.yaml", "--num-frames", 30)
    ok(extract_cli(), "extract", recording, *common, cwd=root)
    ok(extract_cli(), "aggregate-results", cwd=root)
    with h5py.File(session_h5(root), "r") as f:
        first_roi = f["metadata/extraction/roi"][()]

    ok(extract_cli(), "extract", recording, *common, "--bg-roi-index", 1, cwd=root)
    ok(extract_cli(), "aggregate-results", cwd=root)

    assert list(sessions(root)) == [KEY]
    h5 = session_h5(root)
    assert h5s_in(root / "aggregate_results") == [h5]
    with h5py.File(h5, "r") as f:
        assert not np.array_equal(f["metadata/extraction/roi"][()], first_roi)
    assert not any(h5.parent.glob("results_0*"))


# ------------------------------------------------------------ metadata.json


def _acquisition(h5):
    def text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(h5, "r") as f:
        return {name: text(f[f"metadata/acquisition/{name}"][()]) for name in ("SubjectName", "SessionName")}


@pytest.mark.parametrize("front_end", ["notebook", "cli"])
def test_a_recording_without_metadata_gets_one_subject_name_on_both_front_ends(tmp_path, raw_sessions, front_end):
    """No metadata.json beside the recording: the subject name is empty and the session name is the
    folder's name, the same on both front ends, and nothing is written into the recording's folder."""
    root = new_project(tmp_path / "project", raw_sessions, (RECORDINGS[0],), metadata=False)
    folder = root / RECORDINGS[0]
    before = listing(folder)
    if front_end == "notebook":
        from moseq2_app.project import start
        from moseq2_app.util import update_config
        from moseq2_extract.gui import extract_found_sessions

        start(str(root))
        with update_config(str(root / "config.yaml"), section="extract") as extract:
            extract["num_frames"] = 30
        extract_found_sessions(str(root), str(root / "config.yaml"), extract_all=True, skip_extracted=True)
    else:
        ok(extract_cli(), "extract", folder / "depth.avi", "--config-file", root / "config.yaml",
           "--num-frames", 30, cwd=root)

    assert listing(folder) == before
    (h5,) = [p for p in h5s_in(root)]
    assert _acquisition(h5) == {"SubjectName": "", "SessionName": RECORDINGS[0]}


# ------------------------------------------------------------ the notebook's chain, in project mode


@pytest.fixture(scope="module")
def notebook_chain(tmp_path_factory, raw_sessions):
    """The Extract notebook's cells on two recordings: project, config, the arena tool (Save and move on,
    for each), extraction, QC, the preview list, aggregation; then one recording again with ``extract``.

    Returns the project folder and the listing of each recording's folder before the first cell.
    """
    import panel as pn

    from moseq2_app.gui.progress import get_session_paths
    from moseq2_app.main import validate_extractions
    from moseq2_app.project import start
    from moseq2_app.roi.widget import ArenaMaskWidget
    from moseq2_app.util import update_config
    from moseq2_extract.gui import aggregate_extract_results_command, extract_found_sessions

    from widgets import buttons_of, click

    root = new_project(tmp_path_factory.mktemp("chain") / "project", raw_sessions)
    before = {name: listing(root / name) for name in RECORDINGS}
    start(str(root))
    with update_config(str(root / "config.yaml"), section="extract") as extract:
        extract["num_frames"] = 60

    pn.extension()
    widget = ArenaMaskWidget(str(root))
    buttons = buttons_of(widget.view)
    for _ in widget.sessions:
        click(buttons["Save session parameters and move to next"])

    extract_found_sessions(str(root), str(root / "config.yaml"), extract_all=True, skip_extracted=True)
    validate_extractions(str(root))
    previews = get_session_paths(str(root), extracted=True)
    aggregate_extract_results_command(str(root))
    ok(extract_cli(), "extract", root / KEY, "--config-file", root / "config.yaml", "--num-frames", 30, cwd=root)
    return {"root": root, "before": before, "previews": previews}


def test_no_file_is_written_beside_or_into_a_recording(notebook_chain):
    root = notebook_chain["root"]
    for name, before in notebook_chain["before"].items():
        assert listing(root / name) == before, name


def test_previews_leave_no_copies(notebook_chain):
    """Each session's preview movie is one file, in the session's folder, and the preview cell lists it
    (1.5.0 kept one in ``proc/`` and a copy in ``aggregate_results/``)."""
    root = notebook_chain["root"]
    movies = sorted(root.rglob("*.mp4"))
    assert len(movies) == len(RECORDINGS), movies
    assert all(movie.parent.parent == root / "aggregate_results" for movie in movies)
    assert sorted(Path(p) for p in notebook_chain["previews"].values()) == movies
    for key in (f"{name}/depth.avi" for name in RECORDINGS):
        assert session_h5(root, key).parent in {movie.parent for movie in movies}


# ------------------------------------------------------------ session folder names


def test_two_recordings_in_one_folder_get_separate_session_folders(tmp_path):
    """Several archives in one folder are a session each, and each session's folder adds the archive's
    name to its folder's."""
    from moseq2_common.project import Project
    from moseq2_extract.recordings import session_folder_name, session_outputs

    keys = ["cohort1/session_1.tar.gz", "cohort1/session_2.tar.gz", "cohort2/mouse/depth.avi"]
    assert [session_folder_name(key, keys) for key in keys] == [
        "cohort1_session_1", "cohort1_session_2", "cohort2_mouse"]

    root = tmp_path / "project"
    (root / "cohort1").mkdir(parents=True)
    for key in keys[:2]:
        (root / key).write_bytes(b"")
    (root / "config.yaml").write_text("extract: {}\n")
    project = Project.at(root)
    folders = [session_outputs(project, key).folder for key in keys[:2]]
    assert folders == [root / "aggregate_results" / "cohort1_session_1",
                       root / "aggregate_results" / "cohort1_session_2"]
    assert [session_outputs(project, keys[0]).h5.name, session_outputs(project, keys[0]).movie.name] == [
        "results.h5", "results.mp4"]


def test_flattened_names_that_collide_stay_apart():
    """``a_b/c`` and ``a/b_c`` flatten to the same name; each gets a short hash of its key, and the
    names stay the same whatever order the keys come in."""
    from moseq2_extract.recordings import session_folder_name

    keys = ["a_b/c/depth.avi", "a/b_c/depth.avi", "d/depth.avi"]
    names = [session_folder_name(key, keys) for key in keys]
    assert names[0] != names[1]
    assert all(name.startswith("a_b_c_") and len(name) == len("a_b_c_") + 6 for name in names[:2])
    assert names[2] == "d"
    assert [session_folder_name(key, list(reversed(keys))) for key in keys] == names

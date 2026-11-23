"""Which files in a project are its recordings.

Three places look for a project's recordings: the progress cell
(``get_session_paths``), the arena tool (``get_sessions``) and extraction
(``moseq2_extract.project.extract``, which the notebook's Extract step and
``moseq2 extract`` run). All three go through one finder, ``find_recordings``
in ``moseq2_extract.recordings``, so each test here builds a project of empty
files and asks every caller what it found. Nothing is opened, so empty files
are enough. Extraction takes the extensions and the depth to search from the
project's ``config.yaml`` and reports what it found on stderr; the progress
cell and the arena tool take them as arguments and print to stdout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from moseq2_app.gui.progress import get_session_paths, get_sessions
from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml, write_config
from moseq2_extract.gui import generate_config_command

#: Every extension each caller is asked for here, so that the callers differ
#: only in how they search.
ALL = ("dat", "avi", "mkv", "tar.gz")

#: Where each caller writes what it found.
STREAM = {"progress cell": "out", "arena tool": "out", "extraction": "err"}

#: The two messages a user saw for a project the callers found nothing in.
OLD_EMPTY = (
    "Extraction Progress: 0 out of 0 session(s) extracted",
    "No sessions to show. There are either no sessions present or they are all extracted.",
)


def session(root, *parts, recording="depth.dat", metadata=True, extra=()):
    """A session folder at ``root/parts`` holding one recording."""
    folder = Path(root, *parts)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / recording).touch()
    if metadata:
        (folder / "metadata.json").write_text("{}")
    for name in extra:
        (folder / name).touch()
    return folder / recording


def project_config(root, **extract):
    """``root/config.yaml`` as ``generate-config`` writes it, with ``extract`` set in its extract section
    and a head/tail classifier named, so nothing is downloaded."""
    config = Path(root) / "config.yaml"
    if not config.exists():
        generate_config_command(str(config), camera_type="k2")
    data = read_yaml(str(config))
    data["extract"].update({"flip_classifier": str(Path(root) / "flip_classifier.npz"), **extract})
    write_config(data, str(config))
    return config


@pytest.fixture
def chosen(monkeypatch):
    """The recordings extraction extracts, which extracts none of them."""
    recorded = []
    monkeypatch.setattr("moseq2_extract.project.extract_recording",
                        lambda project, input_file, opts, **kw: recorded.append(input_file))
    return recorded


@pytest.fixture
def callers(chosen):
    """The three callers, each returning the recordings it found as sorted paths.

    Extraction is stopped before it extracts anything: the list it would have
    extracted is what it found.
    """
    from moseq2_extract.project import extract

    def progress(root, exts=ALL, **kw):
        return sorted(get_session_paths(str(root), exts=list(exts), **kw).values())

    def arena(root, exts=ALL, **kw):
        return sorted(get_sessions(str(root), skip_extracted=False, extensions=tuple(exts), **kw))

    def extraction(root, exts=ALL, max_depth=5):
        chosen.clear()
        project_config(root, extensions=[f".{e}" for e in exts], max_depth=max_depth)
        extract(Project.at(root))
        return sorted(chosen)

    return {"progress cell": progress, "arena tool": arena, "extraction": extraction}


def found_by_each(callers, root, **kw):
    return {name: find(root, **kw) for name, find in callers.items()}


def test_recordings_one_and_two_levels_down_are_found_by_all_three(callers, tmp_path):
    """A project grouped by condition keeps its sessions two levels down
    (``GroupC/mouse21_session/depth.dat``, the layout in moseq2-app#289)."""
    root = tmp_path / "project"
    expected = sorted(str(p) for p in (
        session(root, "s0"),
        session(root, "s1", recording="depth.avi"),
        session(root, "GroupC", "mouse21_session"),
        session(root, "GroupC", "mouse22_session"),
    ))

    assert found_by_each(callers, root) == {name: expected for name in callers}


def test_deeper_than_the_limit_is_not_found_and_the_message_says_how_to_look_deeper(callers, tmp_path, capsys):
    """Five levels below the project is the default limit. A recording six
    levels down is not found, and the message names the folder whose
    subfolders were not searched and how to search them."""
    root = tmp_path / "project"
    shallow = session(root, "a", "b", "c", "d", "e")
    deep = session(root, "a", "b", "c", "d", "e", "f")
    capsys.readouterr()

    for name, find in callers.items():
        assert find(root) == [str(shallow)], name
        out = getattr(capsys.readouterr(), STREAM[name])
        assert (
            f"{shallow.parent}: its subfolders are more than 5 levels below {root} and were not searched"
            in out
        ), name
        assert "To search deeper, pass a larger max_depth, for example max_depth=10" in out, name

    for name, find in callers.items():
        assert find(root, max_depth=6) == sorted([str(shallow), str(deep)]), name


def test_moseq2_extract_searches_deeper_when_asked(tmp_path, chosen):
    """On the command line the limit is ``--max-depth``."""
    from moseq2_app.cli import cli

    root = tmp_path / "project"
    deep = session(root, "a", "b", "c", "d", "e", "f")
    project_config(root)

    before = os.getcwd()
    os.chdir(root)
    try:
        result = CliRunner().invoke(cli, ["extract"], prog_name="moseq2")
        assert "No recordings found under" in result.stderr
        assert "--max-depth 10" in result.stderr
        assert chosen == []

        result = CliRunner().invoke(cli, ["extract", "--max-depth", "10"], prog_name="moseq2")
    finally:
        os.chdir(before)
    assert result.exit_code == 0, result.output
    assert chosen == [str(deep)]


@pytest.mark.parametrize("caller", ["progress cell", "arena tool", "extraction"])
def test_a_folder_without_metadata_is_found_with_a_warning(callers, tmp_path, capsys, caller):
    """Extraction writes a default metadata.json when there is none, so the
    recording is found; the warning names the folder."""
    root = tmp_path / "project"
    with_metadata = session(root, "s0")
    without = session(root, "s1", metadata=False)

    assert callers[caller](root) == sorted([str(with_metadata), str(without)])
    out = getattr(capsys.readouterr(), STREAM[caller])
    assert f"Warning: {without.parent} has no metadata.json." in out
    assert f"Warning: {with_metadata.parent}" not in out


def test_ir_and_nidaq_files_beside_the_depth_recording_are_left_out(callers, tmp_path):
    root = tmp_path / "project"
    expected = sorted(str(p) for p in (
        session(root, "kinect", extra=("ir.dat", "nidaq.dat")),
        session(root, "converted", recording="depth.avi", extra=("ir.avi",)),
    ))

    assert found_by_each(callers, root) == {name: expected for name in callers}


def test_an_mkv_is_a_recording_whatever_its_name(callers, tmp_path):
    """The Azure and Femto acquisition software names its .mkv as it likes."""
    root = tmp_path / "project"
    expected = [str(session(root, "azure_0", recording="2024-05-01-10-31-07.mkv"))]

    assert found_by_each(callers, root) == {name: expected for name in callers}


def test_a_symlinked_folder_is_searched_and_a_loop_ends(callers, tmp_path):
    """People link their data into the project. A link back up the tree is
    followed only as far as the limit, and each recording is listed once."""
    root = tmp_path / "project"
    elsewhere = tmp_path / "storage"
    session(elsewhere, "s0")
    root.mkdir()
    os.symlink(elsewhere, root / "data")
    os.symlink(root, root / "data" / "loop")

    expected = [str(root / "data" / "s0" / "depth.dat")]
    assert found_by_each(callers, root) == {name: expected for name in callers}


def test_an_empty_project_says_where_it_looked_and_what_counts(tmp_path, capsys):
    """The progress cell and the arena tool used to report an empty project as
    "0 out of 0" and "No sessions to show"; extraction said nothing at all."""
    from moseq2_extract.project import extract

    root = tmp_path / "project"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "rig.txt").touch()
    project_config(root)
    capsys.readouterr()

    from moseq2_app.project import start

    start(str(root))
    progress_out = capsys.readouterr().out

    from moseq2_app.roi.widget import ArenaMaskWidget

    widget = ArenaMaskWidget(str(root))
    assert widget.view is None
    arena_out = capsys.readouterr().out

    extract(Project.at(root))
    extract_out = capsys.readouterr().err

    for out in (progress_out, arena_out, extract_out):
        for old in OLD_EMPTY:
            assert old not in out
        assert f"No recordings found under {root}." in out
        assert f"Searched {root} and its folders down to 5 levels below it, following symlinked folders." in out
        assert "IR videos (ir.dat, ir.avi), the NI-DAQ stream (nidaq.dat) and flipped movies" in out
        assert "A recording is a file ending in .dat, .mkv, .avi or .tar.gz" in out


def test_a_recording_of_a_type_not_asked_for_is_named(callers, tmp_path, capsys):
    """An ``extensions`` setting of .avi, .dat and .tar.gz passes an Azure .mkv
    over, and the message says why."""
    root = tmp_path / "project"
    mkv = session(root, "azure_0", recording="recording.mkv")
    capsys.readouterr()

    assert callers["extraction"](root, exts=("avi", "dat", "tar.gz")) == []
    out = capsys.readouterr().err
    assert f"{mkv.parent}: holds recording.mkv, but this search looks only for .avi, .dat, .tar.gz" in out

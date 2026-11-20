"""Extraction's file layer runs on the files it is given and knows nothing of projects.

``moseq2_extract.steps`` takes a recording, an output folder and an options
object. Run on a recording that sits inside a project folder, from that folder,
it still reads none of the project's files and writes nothing but its outputs:
the recording here sits in a trap project (``tests/layers.py``) whose files would
change the outputs if a step read them. Writing a config file and downloading a
flip classifier write the file they are given and nothing else, and the file
layer never asks a person anything: a choice it needs is a ``ChoiceNeeded``
raised before any work.
"""

from __future__ import annotations

import ast
import inspect
import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

from fixtures import fetch_raw_sessions
from layers import TRAP_UUID, listing, trap_project

from moseq2_common.options import read_settings
from moseq2_common.project import ChoiceNeeded
from moseq2_common.records import read_record
from moseq2_extract import steps
from moseq2_extract.options import EXTRACT_STAGE, ExtractOptions, RoiOptions

SESSION = "saline_example_0"
FLIP_URL = "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.npz"
EXTRACT_PACKAGE = Path(steps.__file__).parent


def call_generate_config(output_file, camera_type):
    """``generate_config`` writing ``output_file``, through the file layer."""
    return steps.generate_config(output_file, camera_type, overwrite=True)


def call_download_flip_file(output_dir, selection=None):
    """``download_flip_file`` into ``output_dir``, through the file layer."""
    return steps.download_flip_file(output_dir, selection)


def copy_recording(raw_sessions, folder: Path) -> Path:
    """A copy of the fixture recording ``SESSION`` in ``folder``/``SESSION``; its ``depth.avi``."""
    session = folder / SESSION
    session.mkdir(parents=True)
    for name in ("depth.avi", "depth_ts.txt", "metadata.json"):
        shutil.copy(Path(raw_sessions) / SESSION / name, session / name)
    return session / "depth.avi"


def kinect_options(folder: Path, cls):
    """``cls`` resolved from a Kinect ``config.yaml`` written in ``folder``, which is no project of the test's."""
    folder.mkdir(parents=True)
    config = folder / "config.yaml"
    steps.generate_config(str(config), "kinect", overwrite=True)
    return cls.resolve(config=read_settings(config, cls, stage_classes=EXTRACT_STAGE))


def test_find_roi_on_plain_files_reads_and_writes_no_project_file(tmp_path, monkeypatch):
    opts = kinect_options(tmp_path / "elsewhere", RoiOptions)
    trap_folder = tmp_path / "trap"
    trap_folder.mkdir()
    recording = copy_recording(fetch_raw_sessions(), trap_folder)
    key = f"{SESSION}/depth.avi"
    trap = trap_project(trap_folder, entries={key: {"noise_tolerance": 1, "bg_roi_index": 3}},
                        sessions={key: "aggregate_results/listed_by_the_trap"})
    before = listing(trap_folder)
    monkeypatch.chdir(trap_folder)

    result = steps.find_roi(str(recording), None, opts)

    proc = recording.parent / "proc"
    assert result.roi_files == [proc / "roi_00.tiff"]
    assert (proc / "bground.tiff").is_file()
    record = read_record(proc / "roi_00.tiff")
    assert record["project"] is None
    assert [item["relative_path"] for item in record["inputs"]] == [None]
    assert [item["path"] for item in record["inputs"]] == [str(recording)]
    assert record["options"]["noise_tolerance"] == opts.noise_tolerance
    assert trap.unchanged()
    new = listing(trap_folder) - before
    assert new and all(path == proc or proc in path.parents for path in new), sorted(new)


def test_find_roi_takes_no_project():
    assert "project" not in inspect.signature(steps.find_roi).parameters
    assert "exclude" not in inspect.signature(steps.find_roi).parameters


def test_extract_on_plain_files_reads_and_writes_no_project_file(tmp_path, monkeypatch):
    opts = replace(kinect_options(tmp_path / "elsewhere", ExtractOptions), num_frames=20)
    trap_folder = tmp_path / "trap"
    trap_folder.mkdir()
    recording = copy_recording(fetch_raw_sessions(), trap_folder)
    key = f"{SESSION}/depth.avi"
    trap = trap_project(trap_folder, entries={key: {"noise_tolerance": 1, "bg_roi_index": 3}},
                        sessions={key: "aggregate_results/listed_by_the_trap"})
    before = listing(trap_folder)
    monkeypatch.chdir(trap_folder)

    result = steps.extract(str(recording), None, opts)

    proc = recording.parent / "proc"
    assert result.h5_file == proc / "results.h5"
    with h5py.File(result.h5_file, "r") as h5:
        uuid = h5["metadata/uuid"][()]
    assert (uuid.decode() if isinstance(uuid, bytes) else str(uuid)) != TRAP_UUID
    record = read_record(result.h5_file)
    assert record["project"] is None
    assert all(item["relative_path"] is None for item in record["inputs"])
    assert str(trap.config_file) not in [item["path"] for item in record["inputs"]]
    assert record["options"]["noise_tolerance"] == opts.noise_tolerance
    assert trap.unchanged()
    new = listing(trap_folder) - before
    assert new and all(path == proc or proc in path.parents for path in new), sorted(new)


def test_generate_config_writes_only_the_given_file(tmp_path):
    """Beside a flat 1.5.0 ``config.yaml``, the file layer writes the file it is given and nothing else; stopping
    on a 1.5.0 project is the project layer's."""
    from moseq2_common.yaml import write_yaml

    write_yaml({"bg_roi_depth_range": [650, 750], "chunk_size": 1000, "camera_type": "k2"},
               str(tmp_path / "config.yaml"))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}

    written = call_generate_config(str(tmp_path / "new.yaml"), "kinect")

    assert Path(written) == tmp_path / "new.yaml"
    assert set(tmp_path.iterdir()) == {*before, tmp_path / "new.yaml"}
    assert all(path.read_bytes() == data for path, data in before.items())
    assert "extract" in (tmp_path / "new.yaml").read_text()


@pytest.fixture
def downloads(monkeypatch):
    """Each download asked for, as (url, file); the file gets a few placeholder bytes."""
    calls = []

    def urlretrieve(url, filename, *args, **kwargs):
        calls.append((url, filename))
        Path(filename).write_bytes(b"npz")
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)
    return calls


def test_download_flip_file_writes_only_the_classifier(tmp_path, monkeypatch, downloads):
    trap = trap_project(tmp_path / "trap")
    before = listing(trap.folder)
    monkeypatch.chdir(trap.folder)

    written = call_download_flip_file(str(trap.folder / "flips"), 1)

    expected = trap.folder / "flips" / Path(FLIP_URL).name
    assert Path(written) == expected
    assert downloads == [(FLIP_URL, str(expected))]
    assert listing(trap.folder) - before == {expected.parent, expected}
    assert trap.unchanged()


def test_download_flip_file_without_a_selection_raises_before_any_download(tmp_path, monkeypatch, capsys,
                                                                           downloads):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("the file layer asked"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ChoiceNeeded) as decision:
        call_download_flip_file(str(tmp_path / "flips"))

    assert decision.value.choices == steps.FLIP_CLASSIFIERS
    assert list(steps.FLIP_CLASSIFIERS) == ["large mice with fibers (K2)", "adult male c57s (K2)",
                                            "mice with Inscopix cables (K2)", "adult male c57s (Azure)"]
    assert decision.value.keyword == "selection"
    assert downloads == []
    assert not (tmp_path / "flips").exists()
    assert capsys.readouterr().out == ""


#: What the file layer must not reach: the project's files and the functions that read or write them.
PROJECT_NAMES = {"find_project", "load", "update", "session_index", "session_key", "read_config", "read_settings",
                 "session_outputs", "project_keys", "extracted_in", "check_not_old"}


def _imported_names(tree):
    """Each name a module imports, and each module it imports from."""
    names, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return names, modules


def _public_functions(tree):
    return [node for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")]


def test_the_file_layer_reaches_no_project_file():
    tree = ast.parse((EXTRACT_PACKAGE / "steps.py").read_text())
    names, modules = _imported_names(tree)

    assert names & PROJECT_NAMES == set()
    assert "moseq2_extract.session_config" not in modules
    for function in _public_functions(tree):
        arguments = function.args
        parameters = {a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
        assert not parameters & {"project", "exclude"}, function.name


@pytest.mark.parametrize("module", ["steps.py", "project.py"])
def test_neither_layer_asks(module):
    tree = ast.parse((EXTRACT_PACKAGE / module).read_text())
    asked = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input"]
    assert asked == []

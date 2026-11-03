"""An Azure or Femto project whose ``config.yaml`` says ``k2`` or ``auto``.

Notebooks from 1.4.5 and earlier left ``camera_type: auto`` (the config cell,
whatever camera was chosen) or ``camera_type: k2`` (the PCA cell) in every
project's ``config.yaml``. The notebook no longer rewrites the key, and ``k2``
means the Kinect v2, so such a project would give its Azure recordings the
Kinect's arena weights and formats, and its PCA the Kinect's filters.

Only Azure Kinect and Orbbec Femto cameras write ``.mkv``. So when the config
names the Kinect v2 (``k2``, ``kinect``) or ``auto``:

- the arena tool and extraction give an ``.mkv`` recording the Azure settings,
  with a line when the config said the Kinect;
- the PCA cell picks the Azure filters when every session it trains on was
  recorded as an ``.mkv``, keeps the stated camera and warns when the sessions
  mix ``.mkv`` and other recordings, and is unchanged for Kinect projects.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.helpers.wrappers as extract_wrappers
from moseq2_app.gui.progress import update_progress
import moseq2_app.util as app_util
from moseq2_app.util import update_config
from moseq2_extract.gui import aggregate_extract_results_command, generate_config_command
from moseq2_common.yaml import read_yaml, write_yaml

from cells import code_cell
from fixtures import fetch_azure_mkv, fetch_raw_sessions

NOTEBOOK = "MoSeq2-Extract-Modeling-Notebook.ipynb"
ARENA_CELL = 5
PCA_CELL = 13

AZURE = {"bg_roi_weights": [10, 0.1, 1], "pixel_format": "gray16be", "movie_dtype": ">u2"}
KINECT = {"bg_roi_weights": [1, 0.1, 1], "pixel_format": "gray16le", "movie_dtype": "<u2"}

AZURE_FILTERS = {"gaussfilter_space": [2.25, 1.5], "tailfilter_size": [13, 13]}


def camera_settings(config):
    return {key: list(config[key]) if key == "bg_roi_weights" else config[key] for key in AZURE}


def camera_lines(out):
    return [line for line in out.splitlines() if "camera_type" in line]


def project_as_145_left_it(root, camera_type, recording="azure-mkv"):
    """One recording, and a ``config.yaml`` holding ``camera_type`` as an older notebook left it."""
    session = root / "session_1"
    session.mkdir(parents=True)
    if recording == "azure-mkv":
        shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session / "depth.mkv")
        (session / "metadata.json").write_text(
            json.dumps({"SessionName": "session_1", "SubjectName": "m1", "StartTime": ""})
        )
        recording = session / "depth.mkv"
    else:
        source = Path(fetch_raw_sessions()) / recording
        for name in ("depth.avi", "depth_ts.txt", "metadata.json"):
            shutil.copy(source / name, session / name)
        recording = session / "depth.avi"
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="azure" if recording.suffix == ".mkv" else "k2")
    with update_config(str(config)) as c:
        c["camera_type"] = camera_type
    progress = root / "progress.yaml"
    progress.write_text("{}\n")
    update_progress(str(progress), "base_dir", str(root))
    update_progress(str(progress), "config_file", str(config))
    return recording


def run_arena_cell(root):
    """The notebook's arena cell, as a kernel runs it."""
    progress = root / "progress.yaml"
    namespace = {"progress_filepath": str(progress), "progress_paths": read_yaml(str(progress)),
                 "skip_extracted": True}
    tree = ast.parse(code_cell(NOTEBOOK, ARENA_CELL))
    last = tree.body.pop()
    exec(compile(tree, "<arena cell>", "exec"), namespace)
    eval(compile(ast.Expression(last.value), "<arena cell>", "eval"), namespace)
    return read_yaml(str(root / "session_config.yaml"))["session_1"]


# ------------------------------------------------------------------- arena tool (R4)


@pytest.mark.parametrize("camera_type", ["k2", "kinect"])
def test_the_arena_tool_gives_a_new_mkv_the_azure_settings_and_says_why(tmp_path, capsys, camera_type):
    recording = project_as_145_left_it(tmp_path, camera_type)
    capsys.readouterr()

    entry = run_arena_cell(tmp_path)

    assert camera_settings(entry) == AZURE
    assert camera_lines(capsys.readouterr().out) == [
        f"{recording} is an .mkv recording, which only Azure Kinect and Orbbec Femto cameras "
        f"write, but config.yaml says camera_type: {camera_type} (the Kinect v2). Used the Azure "
        "settings for it: bg_roi_weights [10, 0.1, 1], pixel_format gray16be, movie_dtype >u2. "
        "Set camera_type: azure in config.yaml."
    ]


def test_the_arena_tool_with_auto_gives_an_mkv_the_azure_settings_quietly(tmp_path, capsys):
    project_as_145_left_it(tmp_path, "auto")
    capsys.readouterr()

    entry = run_arena_cell(tmp_path)

    assert camera_settings(entry) == AZURE
    assert camera_lines(capsys.readouterr().out) == []


def test_the_arena_tool_keeps_the_kinect_for_a_kinect_recording(tmp_path, capsys):
    project_as_145_left_it(tmp_path, "k2", recording="saline_example_0")
    capsys.readouterr()

    entry = run_arena_cell(tmp_path)

    assert camera_settings(entry) == KINECT
    assert camera_lines(capsys.readouterr().out) == []


# ------------------------------------------------------------------- extraction (R4)


@pytest.fixture
def detected(monkeypatch):
    """What ``detect_and_set_camera_parameters`` returned inside extraction, per call."""
    calls = []
    original = extract_wrappers.detect_and_set_camera_parameters

    def spy(config_data, input_file=None):
        result = original(config_data, input_file)
        calls.append(dict(result))
        return result

    monkeypatch.setattr(extract_wrappers, "detect_and_set_camera_parameters", spy)
    return calls


@pytest.mark.parametrize("camera_type, line", [("k2", True), ("auto", False)])
def test_extract_gives_an_mkv_the_azure_settings(tmp_path, detected, camera_type, line):
    recording = project_as_145_left_it(tmp_path, camera_type)

    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(recording), "--config-file", str(tmp_path / "config.yaml"),
         "--output-dir", str(recording.parent / "proc")],
    )

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert [camera_settings(config) for config in detected] == [AZURE]
    printed = camera_lines(result.output)
    if line:
        assert printed == [
            f"{recording} is an .mkv recording, which only Azure Kinect and Orbbec Femto cameras "
            "write, but config.yaml says camera_type: k2 (the Kinect v2). Used the Azure "
            "settings for it: bg_roi_weights [10, 0.1, 1], pixel_format gray16be, movie_dtype >u2. "
            "Set camera_type: azure in config.yaml."
        ]
    else:
        assert printed == []


# ------------------------------------------------------------------------- PCA (B2)


def fake_extracted_session(root, name, recording):
    """A session folder with a recording and a finished extraction, and its aggregated copy.

    Only what the PCA cell's camera decision reads: the recording's name, the
    extraction's results_00.h5 and .yaml beside it, and the copies in
    aggregate_results sharing the extraction's uuid.
    """
    session = root / name
    (session / "proc").mkdir(parents=True)
    (session / recording).write_bytes(b"")
    (session / "metadata.json").write_text("{}")
    uuid = f"uuid-{name}"
    aggregated = root / "aggregate_results"
    aggregated.mkdir(exist_ok=True)
    for stem in (session / "proc" / "results_00", aggregated / f"{name}_results_00"):
        with h5py.File(f"{stem}.h5", "w") as f:
            f["frames"] = np.zeros((2, 4, 4), dtype="uint8")
        write_yaml({"complete": True, "skip": False, "uuid": uuid}, f"{stem}.yaml")


def pca_project(root, camera_type, recordings):
    for i, recording in enumerate(recordings):
        fake_extracted_session(root, f"session_{i}", recording)
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    with update_config(str(config)) as c:
        c["camera_type"] = camera_type
    return config


@pytest.fixture
def run_pca_cell(monkeypatch):
    """Run the notebook's PCA cell with training replaced by a recorder; return the config it trains with."""
    import moseq2_pca.gui

    trained = []
    monkeypatch.setattr(moseq2_pca.gui, "train_pca_command",
                        lambda paths, *args: trained.append(read_yaml(paths["config_file"])))

    def run(root):
        progress = root / "progress.yaml"
        progress.write_text("{}\n")
        update_progress(str(progress), "base_dir", str(root))
        update_progress(str(progress), "config_file", str(root / "config.yaml"))
        update_progress(str(progress), "train_data_dir", str(root / "aggregate_results") + "/")
        namespace = {"progress_filepath": str(progress), "progress_paths": read_yaml(str(progress))}
        exec(compile(code_cell(NOTEBOOK, PCA_CELL), "<PCA cell>", "exec"), namespace)
        (config,) = trained
        return {key: list(config.get(key) or []) for key in AZURE_FILTERS}

    return run


def azure_extracted_project(root, camera_type):
    """The azure-mkv fixture extracted and aggregated, with ``config.yaml`` saying ``camera_type``."""
    recording = project_as_145_left_it(root, camera_type)
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(recording), "--config-file", str(root / "config.yaml"),
         "--output-dir", str(recording.parent / "proc")],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    # the fixture's 60 frames hold no mouse, so its extracted frames are all zero and the
    # default threshold (a mean above 0) would leave it out
    aggregate_extract_results_command(str(root), "{session_name}", "aggregate_results/", mouse_threshold=-1)


@pytest.mark.parametrize("camera_type", ["k2", "auto"])
def test_the_pca_cell_uses_the_azure_filters_for_mkv_sessions(tmp_path, capsys, run_pca_cell, camera_type):
    azure_extracted_project(tmp_path, camera_type)
    capsys.readouterr()

    filters = run_pca_cell(tmp_path)

    assert filters == AZURE_FILTERS
    assert camera_lines(capsys.readouterr().out) == [
        "PCA uses the Azure Kinect and Orbbec Femto filters set in this cell "
        f"(gaussfilter_space, tailfilter_size): config.yaml says camera_type: {camera_type}, but "
        f"the 1 session in {tmp_path / 'aggregate_results'}/ was recorded as an .mkv, which only "
        "those cameras write. Set camera_type: azure in config.yaml."
    ]


def test_the_pca_cell_keeps_the_kinect_filters_for_kinect_sessions(tmp_path, capsys, run_pca_cell):
    pca_project(tmp_path, "k2", ["depth.dat", "depth.avi", "session_20190320.tar.gz"])
    capsys.readouterr()

    filters = run_pca_cell(tmp_path)

    assert filters["gaussfilter_space"] == [1.5, 1]
    assert filters["tailfilter_size"] != [13, 13]
    assert camera_lines(capsys.readouterr().out) == []


@pytest.mark.parametrize("camera_type", ["k2", "auto"])
def test_the_pca_cell_keeps_the_stated_camera_for_mixed_sessions_and_warns(tmp_path, capsys, run_pca_cell,
                                                                          camera_type):
    pca_project(tmp_path, camera_type, ["depth.mkv", "depth.mkv", "depth.dat"])
    capsys.readouterr()

    with pytest.warns(UserWarning) as warned:
        filters = run_pca_cell(tmp_path)

    assert filters["gaussfilter_space"] == [1.5, 1]
    messages = [str(w.message) for w in warned if "camera_type" in str(w.message)]
    assert messages == [
        f"Not every session in {tmp_path / 'aggregate_results'}/ was recorded as an .mkv: 2 were "
        "recorded as .mkv (an Azure Kinect or Orbbec Femto) and 1 as another file type "
        "(.dat, .avi or .tar.gz). "
        f"PCA uses the filters for config.yaml's camera_type: {camera_type}, which suit the Kinect v2. "
        "If every session is from an Azure Kinect or Orbbec Femto, set camera_type: azure in "
        "config.yaml and run this cell again."
    ]


def test_azure_in_the_config_is_azure_whatever_the_sessions(tmp_path):
    pca_project(tmp_path, "azure", ["depth.dat"])
    assert app_util.camera_for_pca("azure", str(tmp_path / "aggregate_results"), str(tmp_path)) == "azure"


def test_a_session_whose_recording_is_gone_is_counted_as_not_found(tmp_path):
    pca_project(tmp_path, "k2", ["depth.mkv", "depth.mkv"])
    shutil.rmtree(tmp_path / "session_1")
    aggregated = tmp_path / "aggregate_results"

    with pytest.warns(UserWarning) as warned:
        camera = app_util.camera_for_pca("k2", str(aggregated), str(tmp_path))

    assert camera == "k2"
    assert [str(w.message) for w in warned if "camera_type" in str(w.message)] == [
        f"Not every session in {aggregated} was recorded as an .mkv: 1 was recorded as .mkv "
        f"(an Azure Kinect or Orbbec Femto) and for 1 the recording could not be found under "
        f"{tmp_path}. PCA uses the filters for config.yaml's camera_type: k2, which suit the "
        "Kinect v2. If every session is from an Azure Kinect or Orbbec Femto, set camera_type: "
        "azure in config.yaml and run this cell again."
    ]

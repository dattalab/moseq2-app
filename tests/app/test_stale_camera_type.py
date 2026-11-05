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
- PCA trains with the filters config.yaml's ``pca`` section holds, which
  ``generate-config`` writes for the camera; nothing infers them at run time.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.helpers.wrappers as extract_wrappers
from moseq2_app.gui.progress import update_progress
import moseq2_app.util as app_util
from moseq2_app.util import update_config
from moseq2_extract.gui import generate_config_command
from moseq2_common.yaml import read_yaml

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
    widget = eval(compile(ast.Expression(last.value), "<arena cell>", "eval"), namespace)
    # what the tool shows for the recording; opening it saves nothing
    (key,) = [key for key in widget.session_config if key.startswith("session_1/")]
    return widget.session_config[key]


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


@pytest.fixture
def run_pca_cell(monkeypatch):
    """Run the notebook's PCA cell with training replaced by a recorder; return the pca section it trains with."""
    import moseq2_pca.gui

    trained = []
    monkeypatch.setattr(moseq2_pca.gui, "train_pca_command",
                        lambda paths, *args: trained.append(read_yaml(paths["config_file"])["pca"]))

    def run(root):
        progress = root / "progress.yaml"
        update_progress(str(progress), "train_data_dir", str(root / "aggregate_results") + "/")
        namespace = {"progress_filepath": str(progress), "progress_paths": read_yaml(str(progress))}
        exec(compile(code_cell(NOTEBOOK, PCA_CELL), "<PCA cell>", "exec"), namespace)
        (section,) = trained
        return {key: list(section.get(key) or []) for key in AZURE_FILTERS}

    return run


@pytest.mark.parametrize("camera_type", ["k2", "auto"])
def test_the_pca_cell_trains_with_the_filters_config_yaml_holds(tmp_path, capsys, run_pca_cell, camera_type):
    """generate-config wrote the Azure filters into the pca section; the cell neither sets nor replaces them."""
    project_as_145_left_it(tmp_path, camera_type)
    capsys.readouterr()

    filters = run_pca_cell(tmp_path)

    assert filters == AZURE_FILTERS
    assert camera_lines(capsys.readouterr().out) == []


def test_the_notebook_no_longer_infers_the_camera_for_pca():
    """The camera's filters are config.yaml's, written by generate-config: nothing decides them at run time."""
    assert not hasattr(app_util, "camera_for_pca")
    assert "camera_for_pca" not in code_cell(NOTEBOOK, PCA_CELL)

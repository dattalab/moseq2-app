"""A command with nothing to work on stops before any work, saying what it looked for and the ways out.

Run in a folder that is not a project, a step that reads a project's sessions
or extractions finds nothing. It stops before it writes anything (no empty
``_pca/``, no bookkeeping file), with a message naming what it looked for and
where, and the ways out: run it from the project folder, or name the project
(``--config-file``, ``--project-file``), or pass the input on the command line.

A required argument left out gets the same treatment: instead of click's bare
"Missing argument", the message says what the argument is and where it usually
is in a project.
"""

from __future__ import annotations

import h5py
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import write_yaml


@pytest.fixture
def empty(tmp_path, monkeypatch):
    """An empty folder outside any project, as the working folder."""
    folder = tmp_path / "empty"
    folder.mkdir()
    monkeypatch.chdir(folder)
    return folder


def written(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


def run(program, args):
    import importlib

    cli = importlib.import_module(f"moseq2_{program}.cli").cli
    return CliRunner().invoke(cli, args)


def stops(result):
    """The command stopped with a message (click's exit codes 1 and 2), not a traceback."""
    assert result.exit_code in (1, 2), result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output
    assert "Missing argument" not in result.output
    return " ".join(result.output.split())


def check(empty, program, args, expected):
    message = stops(run(program, args))
    for text in expected:
        assert text.format(folder=empty) in message, (text, message)
    assert written(empty) == []


# ---------------------------------------------------------------- moseq2-extract (and print-record)

EXTRACT = [
    ("find-roi", [], ["INPUT_FILE", "depth recording", "session_1/depth.avi"]),
    ("extract", [], ["INPUT_FILE", "depth recording", "session_1/depth.avi"]),
    ("convert-raw-to-avi", [], ["INPUT_FILE", "depth.dat"]),
    ("copy-slice", [], ["INPUT_FILE", "depth recording"]),
    ("batch-extract", [], ["INPUT_FOLDER", "project folder", "left out when run from a project folder"]),
    ("batch-extract", ["."], ["{folder}", "project folder", "--config-file"]),
    ("aggregate-results", [], ["{folder}", "project folder"]),
    ("download-flip-file", [], ["{folder}", "project folder", "config.yaml", "generate-config"]),
    ("print-record", [], ["OUTPUT", "aggregate_results", "session key"]),
]


@pytest.mark.parametrize("command, args, expected", EXTRACT, ids=[" ".join([c, *a]) for c, a, _ in EXTRACT])
def test_extract_commands_with_nothing_to_work_on_say_where_they_looked(empty, command, args, expected):
    check(empty, "extract", [command, *args], expected)


# ---------------------------------------------------------------- moseq2-pca

PCA = [
    ("train-pca", [], ["{folder}", "extractions", "project folder", "--config-file", "--input-dir"]),
    ("apply-pca", [], ["{folder}", "extractions", "project folder", "--config-file", "--input-dir"]),
    ("compute-changepoints", [], ["{folder}", "extractions", "project folder", "--config-file", "--input-dir"]),
    ("clip-scores", [], ["PCA_FILE", "_pca/pca_scores.h5"]),
]


@pytest.mark.parametrize("command, args, expected", PCA, ids=[c for c, *_ in PCA])
def test_pca_commands_with_nothing_to_work_on_say_where_they_looked(empty, command, args, expected):
    check(empty, "pca", [command, *args], expected)


def test_a_typed_input_dir_with_no_extractions_names_it(empty):
    (empty / "extractions").mkdir()

    message = stops(run("pca", ["train-pca", "-i", str(empty / "extractions")]))

    assert f"found no extractions in {empty / 'extractions'}" in message
    assert written(empty) == ["extractions"]


@pytest.mark.parametrize("command, by_hand", [("apply-pca", "--pca-file"), ("compute-changepoints", "--pca-file")])
def test_pca_components_not_found_outside_a_project_name_both_ways_out(empty, command, by_hand):
    import numpy as np

    (empty / "session_1").mkdir()
    with h5py.File(empty / "session_1" / "results.h5", "w") as f:
        f["frames"] = np.zeros((10, 80, 80), dtype="uint8")
        f["timestamps"] = np.arange(10) * 33.3
        f["metadata/uuid"] = "uuid-1"

    message = stops(run("pca", [command]))

    assert f"{empty / '_pca' / 'pca.h5'}, which does not exist" in message
    assert "project folder" in message and "--config-file" in message and by_hand in message
    assert written(empty) == ["session_1", "session_1/results.h5"]


# ---------------------------------------------------------------- moseq2-model

MODEL = [
    ("count-frames", [], ["INPUT_FILE", "PC scores", "_pca/pca_scores.h5"]),
    ("learn-model", [], ["INPUT_FILE", "PC scores", "_pca/pca_scores.h5"]),
    ("learn-model", ["scores.h5"], ["DEST_FILE", "models/model.p"]),
    ("apply-model", [], ["MODEL_FILE", "models/model.p"]),
    ("kappa-scan", [], ["INPUT_FILE", "PC scores", "_pca/pca_scores.h5"]),
]


@pytest.mark.parametrize("command, args, expected", MODEL, ids=[" ".join([c, *a]) for c, a, _ in MODEL])
def test_model_commands_missing_an_argument_say_what_it_is(empty, command, args, expected):
    for arg in args:  # an argument given must exist
        with h5py.File(empty / arg, "w"):
            pass
    message = stops(run("model", [command, *args]))
    for text in expected:
        assert text in message, (text, message)
    assert written(empty) == sorted(args)


# ---------------------------------------------------------------- moseq2-viz

VIZ = [
    ("add-group", ["-v", "x", "-g", "y"], ["{folder}", "project folder", "--project-file"]),
    ("plot-scalar-summary", [], ["{folder}", "project folder", "--project-file"]),
    ("plot-group-position-heatmaps", [], ["{folder}", "project folder", "--project-file"]),
    ("plot-verbose-position-heatmaps", [], ["{folder}", "project folder", "--project-file"]),
    ("make-df", [], ["MODEL_FIT", "models/model.p"]),
    ("make-crowd-movies", [], ["MODEL_PATH", "models/model.p"]),
    ("plot-transition-graph", [], ["MODEL_FIT", "models/model.p"]),
    ("plot-stats", [], ["MODEL_FIT", "models/model.p"]),
    ("get-best-model", [], ["MODEL_DIR", "kappa-scan"]),
]


@pytest.mark.parametrize("command, args, expected", VIZ, ids=[c for c, *_ in VIZ])
def test_viz_commands_with_nothing_to_work_on_say_where_they_looked(empty, command, args, expected):
    check(empty, "viz", [command, *args], expected)


def test_get_best_model_with_no_models_says_where_it_looked(empty):
    (empty / "models").mkdir()
    with h5py.File(empty / "changepoints.h5", "w") as f:
        f["cps"] = [0]

    message = stops(run("viz", ["get-best-model", str(empty / "models"), str(empty / "changepoints.h5"),
                                str(empty / "best")]))

    assert f"found no models in {empty / 'models'}" in message
    assert ".p" in message
    assert written(empty) == ["changepoints.h5", "models"]


# ---------------------------------------------------------------- in a project that holds nothing yet


@pytest.fixture
def new_project(empty):
    """A project with its config.yaml and nothing else, as the working folder."""
    from moseq2_extract.gui import generate_config_command

    generate_config_command(str(empty / "config.yaml"), camera_type="k2")
    return empty


def test_aggregate_results_with_nothing_extracted_writes_no_bookkeeping(new_project):
    (new_project / "session_1").mkdir()
    (new_project / "session_1" / "depth.avi").write_bytes(b"0")

    message = stops(run("extract", ["aggregate-results"]))

    assert f"found 1 recording under {new_project}" in message
    assert "batch-extract" in message
    assert written(new_project) == ["config.yaml", "session_1", "session_1/depth.avi"]


def test_train_pca_in_a_project_with_no_sessions_names_input_dir(new_project):
    message = stops(run("pca", ["train-pca"]))

    assert "aggregate-results" in message
    assert "--input-dir" in message
    assert written(new_project) == ["config.yaml"]


@pytest.mark.parametrize("command", ["plot-scalar-summary", "make-df"])
def test_viz_in_a_project_with_no_sessions_says_to_aggregate(new_project, command):
    args = [command] if command != "make-df" else [command, "model.p"]
    if command == "make-df":
        (new_project / "model.p").write_bytes(b"0")
    before = written(new_project)

    message = stops(run("viz", args))

    assert f"no moseq2-project.yaml in {new_project}" in message
    assert "moseq2-extract aggregate-results" in message
    assert written(new_project) == before


def test_viz_with_no_sessions_listed_says_so(new_project):
    write_yaml({"sessions": {}}, str(new_project / "moseq2-project.yaml"))

    message = stops(run("viz", ["plot-scalar-summary"]))

    assert f"{new_project / 'moseq2-project.yaml'} lists no sessions" in message
    assert written(new_project) == ["config.yaml", "moseq2-project.yaml"]

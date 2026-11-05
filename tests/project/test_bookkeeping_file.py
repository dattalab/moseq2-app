"""``moseq2-project.yaml``: where each step's outputs are, on both front ends.

A project's bookkeeping file sits beside its ``config.yaml`` and lists the
sessions (each session's recording, its file in ``aggregate_results/`` and its
uuid) and each step's current outputs: the PCA components, scores and
changepoints, the fitted models and which one is current. Paths in it are
relative to the project folder, with ``/`` between folders on every system, so a
project moved or copied elsewhere runs its next step.

A step run inside a project records what it wrote there, and the next step
reads it there: the command line's ``apply-pca`` records its scores (1.5.0 left
the index without them, so ``make-df`` stopped), aggregating again keeps the
scores and the groups (1.5.0 rewrote the index), PCA trained under another name
is found by ``apply-pca``, and no step writes ``config.yaml`` (1.5.0's PCA steps
wrote their output paths into it). The viz commands and ``learn-model``,
``apply-model`` and ``kappa-scan`` take their sessions from the project
(``--project-file``, by default the working folder's ``moseq2-project.yaml``),
and a viz command run outside a project stops, naming the folder it looked in.
A kappa-scan fit runs from its job settings file and never writes the
bookkeeping file. The notebook's functions take the project folder and read
the bookkeeping file on every call.
"""

from __future__ import annotations

import os
import shlex
import shutil
from contextlib import contextmanager
from pathlib import Path

import h5py
import joblib
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml

#: The two recordings the command-line project is built from, one per condition.
RECORDINGS = ("saline_example_0", "amphetamine_example_0")

BOOKKEEPING = "moseq2-project.yaml"


@contextmanager
def working_in(folder):
    """Run with ``folder`` as the working folder, as a person running a command from the project does."""
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def run(cli, *args, cwd):
    """Invoke ``cli`` with ``args`` from ``cwd``; the result, whatever its exit code."""
    with working_in(cwd):
        return CliRunner().invoke(cli, [str(arg) for arg in args], catch_exceptions=False)


def ok(cli, *args, cwd):
    result = run(cli, *args, cwd=cwd)
    assert result.exit_code == 0, result.output
    return result


def extract_cli():
    from moseq2_extract.cli import cli

    return cli


def pca_cli():
    from moseq2_pca.cli import cli

    return cli


def model_cli():
    from moseq2_model.cli import cli

    return cli


def viz_cli():
    from moseq2_viz.cli import cli

    return cli


def aggregate(root):
    return ok(extract_cli(), "aggregate-results", "-i", root, "-o", root / "aggregate_results", cwd=root)


@pytest.fixture(scope="module")
def cli_project(tmp_path_factory, raw_sessions):
    """Two recordings taken through the command line: extracted, aggregated, PCA, scores, changepoints, a model.

    Every command runs from the project folder and names its paths, as the
    command-line instructions have a person do. Built once; tests that change
    the project work on a copy (``project_copy``).
    """
    root = tmp_path_factory.mktemp("cli") / "project"
    root.mkdir()
    for name in RECORDINGS:
        shutil.copytree(raw_sessions / name, root / name)
    config = root / "config.yaml"
    ok(extract_cli(), "generate-config", "--output-file", config, cwd=root)
    for name in RECORDINGS:
        ok(extract_cli(), "extract", root / name / "depth.avi", "--config-file", config, cwd=root)
    aggregate(root)
    pca_dir = root / "_pca"
    for command in ("train-pca", "apply-pca", "compute-changepoints"):
        ok(pca_cli(), command, "-i", root / "aggregate_results", "-o", pca_dir, "--config-file", config, cwd=root)
    ok(model_cli(), "learn-model", pca_dir / "pca_scores.h5", root / "models" / "model.p", "--num-iter", "5",
       cwd=root)
    return root


@pytest.fixture
def project_copy(cli_project, tmp_path):
    """A copy of the command-line project to change."""
    root = tmp_path / "project"
    shutil.copytree(cli_project, root, symlinks=True)
    return root


def stub_pca_work(monkeypatch):
    """The PCA work functions replaced by ones that write an empty output and record what they were given.

    Each call is recorded as (step, the options dict it was handed, the file it wrote).
    """
    import moseq2_pca.steps as pca_steps

    calls = []

    def written(output_dir, output_file):
        path = Path(output_dir) / f"{output_file}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w"):
            pass
        return path

    def train(input_dir, config, output_dir, output_file):
        path = written(output_dir, output_file)
        calls.append(("train_pca", dict(config), path))
        return {**config, "pca_file": str(path)}

    def apply(input_dir, config, output_dir, output_file):
        path = written(output_dir, output_file)
        calls.append(("apply_pca", dict(config), path))
        return config, True

    def changepoints(input_dir, config, output_dir, output_file):
        path = written(output_dir, output_file)
        calls.append(("compute_changepoints", dict(config), path))
        return config

    monkeypatch.setattr(pca_steps, "train_pca_wrapper", train)
    monkeypatch.setattr(pca_steps, "apply_pca_wrapper", apply)
    monkeypatch.setattr(pca_steps, "compute_changepoints_wrapper", changepoints)
    return calls


def session_index(root):
    from moseq2_common.bookkeeping import session_index as index_of
    from moseq2_common.project import Project

    return index_of(Project.at(root))


# ------------------------------------------------------------------ the command line, step to step


def test_command_line_apply_pca_reaches_make_df(cli_project, tmp_path):
    """make-df, run in the project with no index, reads the scores apply-pca recorded (L1)."""
    output = tmp_path / "moseq_df.csv"

    result = run(viz_cli(), "make-df", cli_project / "models" / "model.p", "--output-file", output, cwd=cli_project)

    assert result.exit_code == 0, result.output
    import pandas as pd

    df = pd.read_csv(output)
    assert sorted(df["SessionName"].unique()) == sorted(RECORDINGS)
    assert df["labels (original)"].notna().any()


def test_aggregating_again_keeps_the_scores(project_copy):
    aggregate(project_copy)

    index = session_index(project_copy)
    assert Path(index["pca_path"]) == project_copy / "_pca" / "pca_scores.h5"
    assert len(index["files"]) == len(RECORDINGS)


def test_aggregating_again_keeps_groups(project_copy):
    """add-group writes config.yaml's groups; aggregating again leaves them, and the sessions carry them."""
    ok(viz_cli(), "add-group", "-k", "SessionName", "-v", "saline_example_0", "-g", "saline", cwd=project_copy)
    config_before = (project_copy / "config.yaml").read_bytes()

    aggregate(project_copy)

    assert (project_copy / "config.yaml").read_bytes() == config_before
    assert read_yaml(str(project_copy / "config.yaml"))["groups"] == {"saline_example_0/depth.avi": "saline"}
    groups = {f["metadata"]["SessionName"]: f["group"] for f in session_index(project_copy)["files"]}
    assert groups == {"saline_example_0": "saline", "amphetamine_example_0": "default"}


# ------------------------------------------------------------------ PCA's outputs


@pytest.mark.parametrize("front_end", ["cli", "notebook"])
def test_pca_trained_under_another_name_is_found_by_apply(front_end, project_copy, monkeypatch):
    calls = stub_pca_work(monkeypatch)

    if front_end == "cli":
        pca_dir = project_copy / "_pca"
        ok(pca_cli(), "train-pca", "-i", project_copy / "aggregate_results", "-o", pca_dir, "--output-file",
           "mouse_pca", cwd=project_copy)
        ok(pca_cli(), "apply-pca", "-i", project_copy / "aggregate_results", "-o", pca_dir, cwd=project_copy)
    else:
        from moseq2_pca.gui import apply_pca_command, train_pca_command

        train_pca_command(str(project_copy), output_file="mouse_pca")
        apply_pca_command(str(project_copy))

    (_, _, trained), (step, settings, _) = calls
    assert step == "apply_pca"
    assert trained == project_copy / "_pca" / "mouse_pca.h5"
    assert Path(settings["pca_file"]) == trained


@pytest.mark.parametrize("front_end", ["cli", "notebook"])
def test_running_pca_leaves_config_yaml_unchanged(front_end, project_copy, monkeypatch):
    stub_pca_work(monkeypatch)
    config = project_copy / "config.yaml"
    before = config.read_bytes()

    if front_end == "cli":
        for command in ("train-pca", "apply-pca", "compute-changepoints"):
            ok(pca_cli(), command, "-i", project_copy / "aggregate_results", "-o", project_copy / "_pca",
               "--config-file", config, cwd=project_copy)
    else:
        from moseq2_pca.gui import apply_pca_command, compute_changepoints_command, train_pca_command

        train_pca_command(str(project_copy))
        apply_pca_command(str(project_copy))
        compute_changepoints_command(str(project_copy))

    assert config.read_bytes() == before


def test_a_notebook_function_reads_the_bookkeeping_file_each_time(project_copy, monkeypatch):
    """Scores computed on the command line between two notebook calls are what the second call uses."""
    from moseq2_pca.gui import compute_changepoints_command

    calls = stub_pca_work(monkeypatch)

    compute_changepoints_command(str(project_copy))
    ok(pca_cli(), "apply-pca", "-i", project_copy / "aggregate_results", "-o", project_copy / "_pca",
       "--output-file", "rescored", cwd=project_copy)
    compute_changepoints_command(str(project_copy))

    first, _, second = calls
    assert Path(first[1]["pca_file_scores"]) == project_copy / "_pca" / "pca_scores.h5"
    assert Path(second[1]["pca_file_scores"]) == project_copy / "_pca" / "rescored.h5"


def test_bookkeeping_paths_are_stored_relative_with_forward_slashes(cli_project, tmp_path, monkeypatch):
    """Outputs named by absolute paths, in a folder whose name has a space, are stored relative with ``/``.

    On Windows the paths given are the system's own, with backslashes.
    """
    root = tmp_path / "Jane Doe" / "my project"
    shutil.copytree(cli_project, root, symlinks=True)
    stub_pca_work(monkeypatch)

    ok(pca_cli(), "train-pca", "-i", str(root / "aggregate_results"), "-o", str(root / "_pca" / "run 2"),
       cwd=root)

    text = (root / BOOKKEEPING).read_text()
    assert "\\" not in text
    assert str(tmp_path) not in text
    stored = read_yaml(str(root / BOOKKEEPING))
    assert stored["pca"]["components"] == "_pca/run 2/pca.h5"
    assert all("\\" not in session["folder"] and not os.path.isabs(session["folder"])
               for session in stored["sessions"].values())
    assert set(stored["sessions"]) == {f"{name}/depth.avi" for name in RECORDINGS}


def test_a_moved_project_runs_its_next_step(cli_project, tmp_path):
    """Copied elsewhere and the original out of reach, the project's next step finds everything it needs."""
    moved = tmp_path / "elsewhere" / "moved"
    shutil.copytree(cli_project, moved, symlinks=True)
    stored = (moved / BOOKKEEPING).read_text()
    assert str(cli_project) not in stored

    output = tmp_path / "moseq_df.csv"
    result = run(viz_cli(), "make-df", moved / "models" / "model.p", "--output-file", output, cwd=moved)

    assert result.exit_code == 0, result.output
    assert output.is_file()
    index = session_index(moved)
    assert all(Path(f["path"][0]).is_relative_to(moved) for f in index["files"])
    assert Path(index["pca_path"]).is_relative_to(moved)


# ------------------------------------------------------------------ models


def test_a_refit_model_is_what_the_analysis_reads(project_copy):
    """Choosing a model for the analysis copies nothing, so a refit under the same name is what it reads."""
    from moseq2_common.bookkeeping import load, resolve_path
    from moseq2_common.project import Project
    from moseq2_app.util import choose_model
    from moseq2_viz.model.util import parse_model_results

    project = Project.at(project_copy)
    model_file = project_copy / "models" / "model.p"
    choose_model(str(project_copy), "model.p")
    assert [p.name for p in (project_copy / "models").rglob("*.p")] == ["model.p"]

    refit = joblib.load(model_file)
    refit["labels"] = [np.where(labels >= 0, labels + 1, labels) for labels in refit["labels"]]
    joblib.dump(refit, model_file)

    current = resolve_path(project, load(project).models["current"])
    assert current == model_file
    read = parse_model_results(str(current))
    for got, want in zip(read["labels"], refit["labels"]):
        np.testing.assert_array_equal(got, want)


def test_a_kappa_scan_job_writes_no_bookkeeping_file(project_copy, monkeypatch):
    """Each kappa-scan fit runs from its job file, takes the project's sessions, and writes no bookkeeping."""
    import moseq2_model.steps as model_steps

    bookkeeping_before = (project_copy / BOOKKEEPING).read_bytes()
    models = project_copy / "scan"
    ok(model_cli(), "kappa-scan", project_copy / "_pca" / "pca_scores.h5", models, "--n-models", "2",
       "--no-get-cmd", cwd=project_copy)

    fits = []
    monkeypatch.setattr(model_steps, "learn_model_wrapper",
                        lambda input_file, dest_file, config: fits.append(config.get("index")))
    commands = [line for line in (models / "train_out.sh").read_text().splitlines()
                if line.startswith("moseq2-model learn-model ")]
    assert len(commands) == 2
    elsewhere = project_copy.parent
    for line in commands:
        ok(model_cli(), *shlex.split(line)[1:], cwd=elsewhere)

    assert not (models / BOOKKEEPING).exists()
    assert (project_copy / BOOKKEEPING).read_bytes() == bookkeeping_before
    for index in fits:
        assert isinstance(index, dict)
        assert sorted(f["metadata"]["SessionName"] for f in index["files"]) == sorted(RECORDINGS)


# ------------------------------------------------------------------ outside a project


VIZ_COMMANDS = {
    "make-df": ["MODEL"],
    "make-crowd-movies": ["MODEL"],
    "plot-scalar-summary": [],
    "plot-group-position-heatmaps": [],
    "plot-verbose-position-heatmaps": [],
    "plot-transition-graph": ["MODEL"],
    "plot-stats": ["MODEL"],
}


@pytest.mark.parametrize("command", VIZ_COMMANDS)
def test_viz_commands_without_a_project_stop_naming_the_folder(command, tmp_path):
    """W5: the command names the folder it looked in and the option that points it at a project."""
    folder = tmp_path / "not a project"
    folder.mkdir()
    model = tmp_path / "model.p"
    model.touch()
    arguments = [model if argument == "MODEL" else argument for argument in VIZ_COMMANDS[command]]

    result = run(viz_cli(), command, *arguments, cwd=folder)

    assert result.exit_code != 0
    message = " ".join(result.output.split())
    assert str(folder) in message
    assert BOOKKEEPING in message
    assert "--project-file" in message
    assert list(folder.iterdir()) == []

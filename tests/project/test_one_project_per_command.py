"""Every command runs on one project, wherever it is run from.

A command finds its project from ``--project-file`` (the project's
``moseq2-project.yaml``), else from ``--config-file``'s project, else from the
folder it runs in. That one project gives the command its settings (its
``config.yaml``), its sessions and groups, the bookkeeping file it records its
outputs in, and the default places of those outputs: ``_pca/`` for the PCA
commands, ``crowd_movies/`` for ``make-crowd-movies`` and ``moseq_df.csv`` for
``make-df``. The line saying a command runs outside a project is printed only
when there is no project.

Each command here is run from a folder other than the project, pointed at the
project with ``--project-file`` or ``--config-file``, with its step stood in for
by a recorder of what it was given. The project's ``config.yaml`` sets values no
class default has, so a setting that reaches the step can only have come from it.

``find-roi`` follows ``extract``'s rules: in a project, the arena tool's entry
for the recording in ``session_config.yaml`` is laid over ``config.yaml``'s
``extract`` section, and the images go into the session's folder in
``aggregate_results/``; nothing is written beside the recording.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from moseq2_common.yaml import write_yaml

BOOKKEEPING = "moseq2-project.yaml"
SESSION = "cohort/mouse_1/depth.avi"
SESSION_FOLDER = "aggregate_results/cohort_mouse_1"

#: Values no class default has, set in the project's config.yaml.
CROWD_MAX_EXAMPLES = 7
MODEL_NUM_ITER = 7
MODEL_VAR_NAME = "scores_in_project"
PCA_CHUNK_SIZE = 777

STANDALONE = "runs outside a project"


@contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def run(cli, *args, cwd):
    """Invoke ``cli`` with ``args`` from ``cwd``; the result, which must have exit code 0."""
    with working_in(cwd):
        result = CliRunner().invoke(cli, [str(arg) for arg in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def project(tmp_path):
    """A project with one listed session and settings that differ from every class default."""
    root = tmp_path / "project"
    (root / SESSION).parent.mkdir(parents=True)
    (root / SESSION).write_bytes(b"\0")
    write_yaml({
        "camera_type": "auto",
        "extract": {},
        "pca": {"chunk_size": PCA_CHUNK_SIZE},
        "model": {"num_iter": MODEL_NUM_ITER, "var_name": MODEL_VAR_NAME},
        "crowd_movies": {"max_examples": CROWD_MAX_EXAMPLES},
        "groups": {SESSION: "saline"},
    }, str(root / "config.yaml"))
    write_yaml({"sessions": {SESSION: {"recording": SESSION, "folder": SESSION_FOLDER, "uuid": "u1"}},
                "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}}, "applied_models": []},
               str(root / BOOKKEEPING))
    for name in ("model.p", "scores.h5", "changepoints.h5"):
        (root / name).write_bytes(b"\0")
    (root / "models").mkdir()
    return root


@pytest.fixture
def elsewhere(tmp_path):
    """A folder that is not a project, which the commands run from."""
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    return folder


def _called(calls, name):
    def record(*args, **kwargs):
        calls[name] = (args, kwargs)
    return record


# ------------------------------------------------------------------ viz


@pytest.fixture
def viz(monkeypatch):
    import moseq2_viz.cli as cli

    calls = {}
    for name in ("make_crowd_movies", "make_df", "get_best_model"):
        monkeypatch.setattr(cli.steps, name, _called(calls, name))
    return cli.cli, calls


def test_make_crowd_movies_uses_the_project_it_is_pointed_at(project, elsewhere, viz):
    cli, calls = viz
    result = run(cli, "make-crowd-movies", project / "model.p", "--project-file", project / BOOKKEEPING,
                 cwd=elsewhere)

    (index, _, output_dir, opts), _ = calls["make_crowd_movies"]
    assert opts.max_examples == CROWD_MAX_EXAMPLES
    assert [f["key"] for f in index["files"]] == [SESSION]
    assert [f["group"] for f in index["files"]] == ["saline"]
    assert Path(output_dir) == project / "crowd_movies"
    assert STANDALONE not in result.output
    assert list(elsewhere.iterdir()) == []


def test_make_df_writes_into_the_project_it_is_pointed_at(project, elsewhere, viz):
    cli, calls = viz
    result = run(cli, "make-df", project / "model.p", "--project-file", project / BOOKKEEPING, cwd=elsewhere)

    (_, index, output_file), _ = calls["make_df"]
    assert [f["key"] for f in index["files"]] == [SESSION]
    assert Path(output_file) == project / "moseq_df.csv"
    assert STANDALONE not in result.output


def test_get_best_model_records_in_the_project_it_is_pointed_at(project, elsewhere, viz):
    cli, calls = viz
    result = run(cli, "get-best-model", project / "models", project / "changepoints.h5", elsewhere / "best.p",
                 "--project-file", project / BOOKKEEPING, cwd=elsewhere)

    _, kwargs = calls["get_best_model"]
    assert Path(kwargs["project"].folder) == project
    assert STANDALONE not in result.output


# ------------------------------------------------------------------ model


@pytest.fixture
def model(monkeypatch):
    import moseq2_model.cli as cli

    calls = {}
    for name in ("learn_model", "kappa_scan", "apply_model"):
        monkeypatch.setattr(cli.steps, name, _called(calls, name))
    return cli.cli, calls


def _project_of(kwargs):
    return Path(kwargs["project"].folder) if kwargs.get("project") is not None else None


@pytest.mark.parametrize("pointer", ["--project-file", "--config-file"])
def test_learn_model_uses_the_project_it_is_pointed_at(pointer, project, elsewhere, model):
    cli, calls = model
    target = project / (BOOKKEEPING if pointer == "--project-file" else "config.yaml")
    result = run(cli, "learn-model", project / "scores.h5", elsewhere / "model.p", pointer, target, cwd=elsewhere)

    (_, _, opts), kwargs = calls["learn_model"]
    assert opts.num_iter == MODEL_NUM_ITER
    assert _project_of(kwargs) == project
    assert [f["key"] for f in kwargs["index"]["files"]] == [SESSION]
    assert STANDALONE not in result.output


def test_kappa_scan_uses_the_project_it_is_pointed_at(project, elsewhere, model):
    cli, calls = model
    result = run(cli, "kappa-scan", project / "scores.h5", project / "models", "--project-file",
                 project / BOOKKEEPING, cwd=elsewhere)

    (_, _, opts), kwargs = calls["kappa_scan"]
    assert opts.num_iter == MODEL_NUM_ITER
    assert kwargs["learn"].num_iter == MODEL_NUM_ITER
    assert _project_of(kwargs) == project
    assert [f["key"] for f in kwargs["index"]["files"]] == [SESSION]
    assert STANDALONE not in result.output


def test_apply_model_uses_the_project_it_is_pointed_at(project, elsewhere, model):
    cli, calls = model
    result = run(cli, "apply-model", project / "model.p", project / "scores.h5", project / "applied.h5",
                 "--project-file", project / BOOKKEEPING, cwd=elsewhere)

    (_, _, _, opts), kwargs = calls["apply_model"]
    assert opts.var_name == MODEL_VAR_NAME
    assert _project_of(kwargs) == project
    assert [f["key"] for f in kwargs["index"]["files"]] == [SESSION]
    assert STANDALONE not in result.output


# ------------------------------------------------------------------ pca


@pytest.fixture
def pca(monkeypatch):
    import moseq2_pca.cli as cli

    calls = {}
    for name in ("train_pca", "apply_pca", "compute_changepoints"):
        monkeypatch.setattr(cli.steps, name, _called(calls, name))
    return cli.cli, calls


PCA_COMMANDS = {"train-pca": "train_pca", "apply-pca": "apply_pca", "compute-changepoints": "compute_changepoints"}


@pytest.mark.parametrize("command", PCA_COMMANDS)
def test_pca_commands_write_into_the_project_their_config_file_is_in(command, project, elsewhere, pca):
    cli, calls = pca
    result = run(cli, command, "--config-file", project / "config.yaml", cwd=elsewhere)

    (_, output_dir, _, opts), kwargs = calls[PCA_COMMANDS[command]]
    assert opts.chunk_size == PCA_CHUNK_SIZE
    assert Path(output_dir) == project / "_pca"
    assert _project_of(kwargs) == project
    assert STANDALONE not in result.output
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("command", PCA_COMMANDS)
def test_pca_commands_outside_a_project_default_to_the_folder_they_run_in(command, elsewhere, pca):
    """``-o`` and ``-i`` default to the folder the command runs in, not the one the package was imported in."""
    cli, calls = pca
    result = run(cli, command, cwd=elsewhere)

    (input_dir, output_dir, _, _), kwargs = calls[PCA_COMMANDS[command]]
    assert Path(output_dir) == elsewhere / "_pca"
    assert Path(input_dir) == elsewhere
    assert _project_of(kwargs) is None
    assert STANDALONE in result.output


# ------------------------------------------------------------------ find-roi


def test_find_roi_in_a_project_uses_the_arena_entry_and_writes_into_the_session_folder(project, monkeypatch):
    import moseq2_extract.cli as cli
    import moseq2_extract.steps as steps

    write_yaml({"extract": {"noise_tolerance": 20}}, str(project / "config.yaml"))
    write_yaml({SESSION: {"noise_tolerance": 35}}, str(project / "session_config.yaml"))
    received = {}

    def get_roi_wrapper(input_file, config_data, output_dir=None):
        received["noise_tolerance"] = config_data["noise_tolerance"]
        received["output_dir"] = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        config_data["output_dir"] = output_dir

    monkeypatch.setattr(steps, "get_roi_wrapper", get_roi_wrapper)
    recording_folder = (project / SESSION).parent
    before = sorted(recording_folder.iterdir())

    run(cli.cli, "find-roi", project / SESSION, cwd=project)

    assert received["noise_tolerance"] == 35
    assert Path(received["output_dir"]) == project / SESSION_FOLDER
    assert sorted(recording_folder.iterdir()) == before

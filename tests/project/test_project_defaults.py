"""Inside a project, a step's data arguments default from the project, and a typed value overrides them.

Run from a project folder (or pointed at one with ``--project-file`` or
``--config-file``), a command whose input is something the project already holds
can be run without naming it:

- ``learn-model``, ``kappa-scan``, ``apply-model``, ``count-frames`` and
  ``clip-scores`` read the PC scores ``moseq2-project.yaml`` names;
- ``make-df``, ``make-crowd-movies``, ``plot-transition-graph`` and
  ``plot-stats`` read the project's current model;
- ``get-best-model`` reads the changepoints it names, compares the models in
  ``models/`` and draws ``plots/model_vs_pc_changepoints``, as the notebook does;
- ``batch-extract`` searches the project folder;
- ``learn-model`` writes ``models/model.p``, ``kappa-scan`` writes into
  ``models/`` and ``apply-model`` into ``applied_models/``, as the notebook does;
- ``compute-changepoints`` reads the components the project names, or the ones
  given with ``--pca-file``.

A path typed on the command line is used instead, for that run. Outside a
project the arguments stay required, and the message for one left out says it
can be left out when run from a project folder. Each command runs here with its
step stood in for by a recorder of what it was given.
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
SCORES = "_pca/pca_scores.h5"
CHANGEPOINTS = "_pca/changepoints.h5"
COMPONENTS = "_pca/pca.h5"
CURRENT = "models/model.p"


@contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def invoke(cli, *args, cwd):
    with working_in(cwd):
        return CliRunner().invoke(cli, [str(arg) for arg in args])


def run(cli, *args, cwd):
    result = invoke(cli, *args, cwd=cwd)
    assert result.exit_code == 0, result.output
    return result


def stops(result):
    assert result.exit_code in (1, 2), result.output
    assert "Traceback" not in result.output
    return " ".join(result.output.split())


@pytest.fixture
def project(tmp_path):
    """A project whose moseq2-project.yaml names PC scores, components, changepoints and a current model."""
    root = tmp_path / "project"
    (root / SESSION).parent.mkdir(parents=True)
    (root / SESSION).write_bytes(b"\0")
    write_yaml({"camera_type": "auto", "extract": {}, "pca": {}, "model": {}, "crowd_movies": {}},
               str(root / "config.yaml"))
    write_yaml({"sessions": {SESSION: {"recording": SESSION, "folder": "aggregate_results/cohort_mouse_1",
                                       "uuid": "u1"}},
                "pca": {"components": COMPONENTS, "scores": SCORES, "changepoints": CHANGEPOINTS},
                "models": {"fitted": [CURRENT], "current": CURRENT, "analysis": {}}, "applied_models": []},
               str(root / BOOKKEEPING))
    for name in (SCORES, CHANGEPOINTS, COMPONENTS, CURRENT):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"\0")
    return root


@pytest.fixture
def typed(tmp_path):
    """Files outside the project, typed on the command line."""
    folder = tmp_path / "typed"
    folder.mkdir()
    for name in ("scores.h5", "model.p", "changepoints.h5", "pca.h5"):
        (folder / name).write_bytes(b"\0")
    (folder / "models").mkdir()
    return folder


@pytest.fixture
def elsewhere(tmp_path):
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    return folder


def _recorder(calls, name):
    def record(*args, **kwargs):
        calls[name] = (args, kwargs)
    return record


def stand_in(monkeypatch, module, *names):
    calls = {}
    for name in names:
        monkeypatch.setattr(module, name, _recorder(calls, name))
    return calls


# ------------------------------------------------------------------ the PC scores


@pytest.fixture
def model(monkeypatch):
    import moseq2_model.cli as cli

    return cli.cli, stand_in(monkeypatch, cli.steps, "learn_model", "kappa_scan", "apply_model")


def test_learn_model_defaults_to_the_project_scores_and_models_folder(project, model):
    cli, calls = model
    run(cli, "learn-model", cwd=project)

    (input_file, dest_file, _), _ = calls["learn_model"]
    assert Path(input_file) == project / SCORES
    assert Path(dest_file) == project / "models" / "model.p"


def test_learn_model_typed_arguments_win(project, typed, model):
    cli, calls = model
    run(cli, "learn-model", typed / "scores.h5", typed / "fit.p", cwd=project)

    (input_file, dest_file, _), _ = calls["learn_model"]
    assert Path(input_file) == typed / "scores.h5"
    assert Path(dest_file) == typed / "fit.p"


def test_learn_model_with_only_the_scores_typed_writes_models_model_p(project, typed, model):
    cli, calls = model
    run(cli, "learn-model", typed / "scores.h5", cwd=project)

    (input_file, dest_file, _), _ = calls["learn_model"]
    assert Path(input_file) == typed / "scores.h5"
    assert Path(dest_file) == project / "models" / "model.p"


def test_learn_model_pointed_at_a_project_uses_its_scores(project, elsewhere, model):
    cli, calls = model
    run(cli, "learn-model", "--project-file", project / BOOKKEEPING, cwd=elsewhere)

    (input_file, dest_file, _), _ = calls["learn_model"]
    assert Path(input_file) == project / SCORES
    assert Path(dest_file) == project / "models" / "model.p"


def test_kappa_scan_defaults_to_the_project_scores_and_models_folder(project, typed, model):
    cli, calls = model
    run(cli, "kappa-scan", cwd=project)
    (input_file, output_dir, _), _ = calls["kappa_scan"]
    assert Path(input_file) == project / SCORES
    assert Path(output_dir) == project / "models"

    run(cli, "kappa-scan", typed / "scores.h5", typed / "models", cwd=project)
    (input_file, output_dir, _), _ = calls["kappa_scan"]
    assert Path(input_file) == typed / "scores.h5"
    assert Path(output_dir) == typed / "models"


def test_apply_model_defaults_to_the_project_scores_and_applied_models(project, typed, model):
    cli, calls = model
    run(cli, "apply-model", typed / "model.p", cwd=project)
    (model_file, pc_file, dest_file, _), _ = calls["apply_model"]
    assert Path(model_file) == typed / "model.p"
    assert Path(pc_file) == project / SCORES
    assert Path(dest_file) == project / "applied_models" / "model.p"

    run(cli, "apply-model", typed / "model.p", typed / "scores.h5", typed / "applied.p", cwd=project)
    (model_file, pc_file, dest_file, _), _ = calls["apply_model"]
    assert Path(pc_file) == typed / "scores.h5"
    assert Path(dest_file) == typed / "applied.p"


def test_count_frames_defaults_to_the_project_scores(project, typed, monkeypatch):
    import moseq2_model.cli as cli

    calls = stand_in(monkeypatch, cli, "count_frames_wrapper")
    run(cli.cli, "count-frames", cwd=project)
    assert Path(calls["count_frames_wrapper"][1]["input_file"]) == project / SCORES

    run(cli.cli, "count-frames", typed / "scores.h5", cwd=project)
    assert Path(calls["count_frames_wrapper"][1]["input_file"]) == typed / "scores.h5"


def test_clip_scores_defaults_to_the_project_scores(project, typed, monkeypatch):
    import moseq2_pca.cli as cli

    calls = stand_in(monkeypatch, cli, "clip_scores_wrapper")
    run(cli.cli, "clip-scores", "10", cwd=project)
    (pca_file, clip_samples, _), _ = calls["clip_scores_wrapper"]
    assert Path(pca_file) == project / SCORES
    assert clip_samples == 10

    run(cli.cli, "clip-scores", typed / "scores.h5", "20", cwd=project)
    (pca_file, clip_samples, _), _ = calls["clip_scores_wrapper"]
    assert Path(pca_file) == typed / "scores.h5"
    assert clip_samples == 20


def test_a_project_with_no_scores_yet_says_how_to_make_them(project, model):
    cli, _ = model
    write_yaml({"sessions": {}, "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}},
                "applied_models": []}, str(project / BOOKKEEPING))

    message = stops(invoke(cli, "learn-model", cwd=project))

    assert "INPUT_FILE" in message
    assert str(project / BOOKKEEPING) in message
    assert "moseq2-pca apply-pca" in message


def test_a_recorded_file_that_is_gone_is_named(project, model):
    cli, _ = model
    (project / SCORES).unlink()

    message = stops(invoke(cli, "learn-model", cwd=project))

    assert str(project / SCORES) in message
    assert "does not exist" in message


# ------------------------------------------------------------------ the current model


@pytest.fixture
def viz(monkeypatch):
    import moseq2_viz.cli as cli

    return cli.cli, stand_in(monkeypatch, cli.steps, "make_df", "make_crowd_movies", "plot_transition_graph",
                             "plot_stats", "get_best_model")


def _model_of(name, args):
    return {"make_df": args[0], "make_crowd_movies": args[1], "plot_transition_graph": args[1],
            "plot_stats": args[1]}[name]


VIZ_MODEL = {"make-df": "make_df", "make-crowd-movies": "make_crowd_movies",
             "plot-transition-graph": "plot_transition_graph", "plot-stats": "plot_stats"}


@pytest.mark.parametrize("command", VIZ_MODEL)
def test_viz_commands_default_to_the_current_model(command, project, typed, viz):
    cli, calls = viz
    run(cli, command, cwd=project)
    args, _ = calls[VIZ_MODEL[command]]
    assert Path(_model_of(VIZ_MODEL[command], args)) == project / CURRENT

    run(cli, command, typed / "model.p", cwd=project)
    args, _ = calls[VIZ_MODEL[command]]
    assert Path(_model_of(VIZ_MODEL[command], args)) == typed / "model.p"


def test_a_project_with_no_current_model_says_how_to_pick_one(project, viz):
    cli, _ = viz
    write_yaml({"sessions": {SESSION: {"recording": SESSION, "folder": "aggregate_results/cohort_mouse_1",
                                       "uuid": "u1"}},
                "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}}, "applied_models": []},
               str(project / BOOKKEEPING))

    message = stops(invoke(cli, "plot-stats", cwd=project))

    assert "MODEL_FIT" in message
    assert "current model" in message
    assert "learn-model" in message and "get-best-model" in message


def test_get_best_model_defaults_to_the_project(project, typed, viz):
    cli, calls = viz
    run(cli, "get-best-model", cwd=project)
    (model_dir, cp_path, output_file, _), _ = calls["get_best_model"]
    assert Path(model_dir) == project / "models"
    assert Path(cp_path) == project / CHANGEPOINTS
    assert Path(output_file) == project / "plots" / "model_vs_pc_changepoints"

    run(cli, "get-best-model", typed / "models", typed / "changepoints.h5", typed / "best", cwd=project)
    (model_dir, cp_path, output_file, _), _ = calls["get_best_model"]
    assert Path(model_dir) == typed / "models"
    assert Path(cp_path) == typed / "changepoints.h5"
    assert Path(output_file) == typed / "best"


# ------------------------------------------------------------------ batch-extract


def test_batch_extract_defaults_to_the_project_folder(project, typed, monkeypatch):
    import moseq2_extract.cli as cli

    calls = {}

    class Result:
        failures = []

    def batch_extract(input_folder, *args, **kwargs):
        calls["input_folder"] = input_folder
        return Result()

    monkeypatch.setattr(cli.steps, "batch_extract", batch_extract)
    run(cli.cli, "batch-extract", cwd=project)
    assert Path(calls["input_folder"]).resolve() == project.resolve()

    run(cli.cli, "batch-extract", typed, cwd=project)
    assert Path(calls["input_folder"]).resolve() == typed.resolve()


# ------------------------------------------------------------------ compute-changepoints --pca-file


@pytest.fixture
def pca(monkeypatch):
    import moseq2_pca.cli as cli

    return cli.cli, stand_in(monkeypatch, cli.steps, "compute_changepoints")


def test_compute_changepoints_reads_the_project_components(project, pca):
    cli, calls = pca
    run(cli, "compute-changepoints", cwd=project)

    _, kwargs = calls["compute_changepoints"]
    assert Path(kwargs["components_file"]) == project / COMPONENTS


def test_compute_changepoints_pca_file_wins(project, typed, pca):
    cli, calls = pca
    run(cli, "compute-changepoints", "--pca-file", typed / "pca.h5", cwd=project)

    _, kwargs = calls["compute_changepoints"]
    assert Path(kwargs["components_file"]) == typed / "pca.h5"


# ------------------------------------------------------------------ outside a project

OUTSIDE = [
    ("model", ["learn-model"], "INPUT_FILE"),
    ("model", ["kappa-scan"], "INPUT_FILE"),
    ("model", ["count-frames"], "INPUT_FILE"),
    ("pca", ["clip-scores"], "PCA_FILE"),
    ("viz", ["plot-stats"], "MODEL_FIT"),
    ("viz", ["get-best-model"], "MODEL_DIR"),
    ("extract", ["batch-extract"], "INPUT_FOLDER"),
]


@pytest.mark.parametrize("program, args, name", OUTSIDE, ids=[a[0] for _, a, _ in OUTSIDE])
def test_outside_a_project_a_left_out_argument_says_it_can_be_left_out_in_one(program, args, name, elsewhere):
    import importlib

    cli = importlib.import_module(f"moseq2_{program}.cli").cli
    message = stops(invoke(cli, *args, cwd=elsewhere))

    assert name in message
    assert "left out when run from a project folder" in message
    assert list(elsewhere.iterdir()) == []


def test_clip_scores_outside_a_project_with_only_the_count_names_the_scores(elsewhere):
    from moseq2_pca.cli import cli

    message = stops(invoke(cli, "clip-scores", "10", cwd=elsewhere))

    assert "PCA_FILE" in message


# ------------------------------------------------------------------ smaller fixes


def test_download_flip_file_output_dir_is_the_folder_it_runs_in(project, monkeypatch):
    import moseq2_extract.cli as cli

    received = {}
    monkeypatch.setattr(cli.setup, "download_flip_file",
                        lambda config_file, output_dir, selection: received.setdefault("output_dir", output_dir))
    run(cli.cli, "download-flip-file", cwd=project)

    assert Path(received["output_dir"]).resolve() == project.resolve()


def test_plot_scalar_summary_help_names_the_file_it_writes():
    from moseq2_viz.cli import cli

    result = CliRunner().invoke(cli, ["plot-scalar-summary", "--help"])

    assert "plots/scalars_summary" in " ".join(result.output.split())

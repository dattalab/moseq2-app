"""Each analysis step is one function in ``moseq2_viz.steps``, called by the front ends.

Every ``moseq2-viz`` pipeline command resolves its settings into one options
object and hands it to its step, which runs the work; the notebook's
``get_best_fit_model`` calls the same ``get_best_model`` step as
``moseq2-viz get-best-model``.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_viz.cli as viz_cli

#: Each command, its step in ``moseq2_viz.steps``, its options class in
#: ``moseq2_viz.options`` and the files it takes as arguments.
COMMANDS = {
    "add-group": ("add_group", "AddGroupOptions", ("index.yaml",)),
    "get-best-model": ("get_best_model", "BestModelOptions", ("models", "changepoints.h5", "best")),
    "make-df": ("make_df", "MakeDfOptions", ("model.p", "index.yaml")),
    "make-crowd-movies": ("make_crowd_movies", "CrowdMovieOptions", ("index.yaml", "model.p")),
    "plot-scalar-summary": ("plot_scalar_summary", "ScalarSummaryOptions", ("index.yaml",)),
    "plot-group-position-heatmaps": ("plot_group_position_heatmaps", "HeatmapOptions", ("index.yaml",)),
    "plot-verbose-position-heatmaps": ("plot_verbose_position_heatmaps", "HeatmapOptions", ("index.yaml",)),
    "plot-transition-graph": ("plot_transition_graph", "TransitionGraphOptions", ("index.yaml", "model.p")),
    "plot-stats": ("plot_stats", "StatsPlotOptions", ("index.yaml", "model.p")),
}

#: Options a command needs to start. ``add-group`` stops on its own default
#: ``--value`` ("Value must be an iterable."), as 1.5.0's does.
TYPED = {"add-group": ["-v", "Mouse"]}


@pytest.fixture
def steps():
    import moseq2_viz.steps

    return moseq2_viz.steps


@pytest.fixture
def options():
    import moseq2_viz.options

    return moseq2_viz.options


@pytest.fixture
def called(monkeypatch, steps):
    """Each step replaced by a recorder of the options object it is given."""
    calls = {step: [] for step, _, _ in COMMANDS.values()}

    def recorder(step):
        def record(*args, **kwargs):
            calls[step].append(next(a for a in [*args, *kwargs.values()] if hasattr(a, "as_dict")))
        return record

    for step in calls:
        monkeypatch.setattr(steps, step, recorder(step))
    return calls


@pytest.fixture
def files(tmp_path):
    """The argument files, existing (the commands check that they do)."""
    (tmp_path / "models").mkdir()
    for name in ("index.yaml", "model.p", "changepoints.h5"):
        (tmp_path / name).write_text("")
    return tmp_path


@pytest.mark.parametrize("command", COMMANDS)
def test_every_viz_command_calls_its_step(command, files, called, options):
    step, cls, arguments = COMMANDS[command]
    result = CliRunner().invoke(viz_cli.cli, [command, *(str(files / a) for a in arguments), *TYPED.get(command, [])],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(called[step]) == 1
    assert type(called[step][0]) is getattr(options, cls)


def test_get_best_fit_model_calls_the_step(tmp_path, monkeypatch, steps, options):
    from moseq2_viz.gui import get_best_fit_model

    calls = []

    class Figure:
        def show(self, warn=True):
            pass

    def record(model_dir, cp_path, output_file, opts):
        calls.append((model_dir, cp_path, output_file, opts))
        return steps.BestModelResult(best_model_info={"best": "m.p"}, figure=Figure(), figure_files=[])

    monkeypatch.setattr(steps, "get_best_model", record)
    changepoints = tmp_path / "changepoints.h5"
    changepoints.write_text("")
    paths = {"base_model_path": str(tmp_path / "models"), "changepoints_path": str(changepoints),
             "plot_path": str(tmp_path / "plots"), "pca_dirname": str(tmp_path)}

    best = get_best_fit_model(paths, plot_all=True, objective="duration (median match)")

    assert best == {"best": "m.p"}
    assert calls == [(paths["base_model_path"], str(changepoints),
                      str(tmp_path / "plots" / "model_vs_pc_changepoints"),
                      options.BestModelOptions(plot_all=True, objective="duration (median match)"))]

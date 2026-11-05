"""Each analysis step is one function in ``moseq2_viz.steps``, called by the front ends.

Every ``moseq2-viz`` pipeline command resolves its settings into one options
object and hands it to its step, with the sessions of the project
``--project-file`` names, which runs the work; the notebook's
``get_best_fit_model`` calls the same ``get_best_model`` step as
``moseq2-viz get-best-model``, with the changepoints the project's
``moseq2-project.yaml`` names.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_viz.cli as viz_cli

#: Each command, its step in ``moseq2_viz.steps``, its options class in
#: ``moseq2_viz.options`` (None for ``make-df``, which has no settings) and the
#: files it takes as arguments. Every command but ``get-best-model`` is also given
#: ``--project-file``.
COMMANDS = {
    "add-group": ("add_group", "AddGroupOptions", ()),
    "get-best-model": ("get_best_model", "BestModelOptions", ("models", "changepoints.h5", "best")),
    "make-df": ("make_df", None, ("model.p",)),
    "make-crowd-movies": ("make_crowd_movies", "CrowdMovieOptions", ("model.p",)),
    "plot-scalar-summary": ("plot_scalar_summary", "ScalarSummaryOptions", ()),
    "plot-group-position-heatmaps": ("plot_group_position_heatmaps", "HeatmapOptions", ()),
    "plot-verbose-position-heatmaps": ("plot_verbose_position_heatmaps", "HeatmapOptions", ()),
    "plot-transition-graph": ("plot_transition_graph", "TransitionGraphOptions", ("model.p",)),
    "plot-stats": ("plot_stats", "StatsPlotOptions", ("model.p",)),
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
            calls[step].append(next((a for a in [*args, *kwargs.values()] if hasattr(a, "as_dict")), None))
        return record

    for step in calls:
        monkeypatch.setattr(steps, step, recorder(step))
    return calls


@pytest.fixture
def files(tmp_path):
    """The argument files, existing (the commands check that they do), in a project with no sessions yet."""
    from configs import write_bookkeeping

    (tmp_path / "models").mkdir()
    for name in ("model.p", "changepoints.h5"):
        (tmp_path / name).write_text("")
    (tmp_path / "config.yaml").write_text("camera_type: auto\n")
    write_bookkeeping(tmp_path)
    return tmp_path


@pytest.mark.parametrize("command", COMMANDS)
def test_every_viz_command_calls_its_step(command, files, called, options):
    step, cls, arguments = COMMANDS[command]
    project = [] if command == "get-best-model" else ["--project-file", str(files / "moseq2-project.yaml")]
    result = CliRunner().invoke(viz_cli.cli, [command, *(str(files / a) for a in arguments), *project,
                                              *TYPED.get(command, [])], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(called[step]) == 1
    assert type(called[step][0]) is (type(None) if cls is None else getattr(options, cls))


def test_get_best_fit_model_calls_the_step(tmp_path, monkeypatch, steps, options):
    from moseq2_viz.gui import get_best_fit_model

    calls = []

    class Figure:
        def show(self, warn=True):
            pass

    def record(model_dir, cp_path, output_file, opts, **_):
        calls.append((model_dir, cp_path, output_file, opts))
        return steps.BestModelResult(best_model_info={"best": "m.p"}, figure=Figure(), figure_files=[])

    from configs import write_bookkeeping

    monkeypatch.setattr(steps, "get_best_model", record)
    changepoints = tmp_path / "changepoints.h5"
    changepoints.write_text("")
    write_bookkeeping(tmp_path, changepoints="changepoints.h5")

    best = get_best_fit_model(str(tmp_path), plot_all=True, objective="duration (median match)")

    assert best == {"best": "m.p"}
    assert calls == [(str(tmp_path / "models"), str(changepoints),
                      str(tmp_path / "plots" / "model_vs_pc_changepoints"),
                      options.BestModelOptions(plot_all=True, objective="duration (median match)"))]


def test_get_best_fit_model_takes_its_defaults_from_the_options_class(tmp_path, monkeypatch, steps):
    """A default changed in ``BestModelOptions`` reaches the notebook's model selection."""
    import dataclasses

    import moseq2_viz.gui as gui

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Changed(gui.BestModelOptions):
        fps: int = 25
        ext: str = "pkl"

    calls = []

    class Figure:
        def show(self, warn=True):
            pass

    def record(model_dir, cp_path, output_file, opts, **_):
        calls.append(opts)
        return steps.BestModelResult(best_model_info={}, figure=Figure(), figure_files=[])

    from configs import write_bookkeeping

    monkeypatch.setattr(gui, "BestModelOptions", Changed)
    monkeypatch.setattr(steps, "get_best_model", record)
    (tmp_path / "changepoints.h5").write_text("")
    write_bookkeeping(tmp_path, changepoints="changepoints.h5")

    gui.get_best_fit_model(str(tmp_path), plot_all=True)
    gui.get_best_fit_model(str(tmp_path), fps=30)

    assert [(o.plot_all, o.fps, o.ext, o.objective) for o in calls] == [
        (True, 25, "pkl", "duration (mean match)"), (False, 30, "pkl", "duration (mean match)")]


def test_steps_take_no_argument_no_caller_passes(steps, options):
    """``make_crowd_movies`` renders from ``opts.session_names``; ``make_df`` has no settings."""
    import inspect

    assert "sessions" not in inspect.signature(steps.make_crowd_movies).parameters
    assert list(inspect.signature(steps.make_df).parameters) == ["model_fit", "index", "output_file"]
    assert not hasattr(options, "MakeDfOptions")

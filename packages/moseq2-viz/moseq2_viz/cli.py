"""
CLI for visualizing the results.

Each pipeline command's settings are generated from its options class in
``moseq2_viz.options``; the command resolves the settings and calls its step in
``moseq2_viz.steps``.

The commands work on one project (``--project-file``, by default the one in the
folder the command runs in): its sessions, those its ``moseq2-project.yaml``
lists, each with its group from its ``config.yaml``, and its settings, which
for ``make-crowd-movies`` are the ``crowd_movies`` section of that
``config.yaml``, less its ``exclude`` (``--exclude`` and ``--no-exclude``
replace it for one run). Outputs given no path go into the project folder, and
the model a command reads is the project's current model when left out
(``get-best-model``: its changepoints and ``models/``). Without a project the
commands stop, saying where they looked; ``get-best-model`` alone runs outside
one.
"""


import click

from moseq2_common.bookkeeping import session_index
from moseq2_common.options import (
    ProjectGroup,
    click_options,
    command_project,
    PROJECT_CHANGEPOINTS,
    PROJECT_MODEL,
    command_settings,
    exclude_options,
    explicit_values,
    in_project,
    output_folder,
    project_argument,
    project_file_option,
    typed_exclude,
)
from moseq2_common.records import print_record_command
from moseq2_viz import steps
from moseq2_viz.options import (
    CROWD_MOVIE_STAGE,
    AddGroupOptions,
    BestModelOptions,
    CrowdMovieOptions,
    HeatmapOptions,
    ScalarSummaryOptions,
    StatsPlotOptions,
    TransitionGraphOptions,
)


@click.group(cls=ProjectGroup, context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _resolve(ctx, cls):
    """``cls`` resolved from the typed options, and the config.yaml of the project the command runs on."""
    config = command_settings(ctx, cls, stage_classes=CROWD_MOVIE_STAGE if cls is CrowdMovieOptions else (cls,))
    return cls.resolve(config=config, explicit=explicit_values(ctx, cls))


_PROJECT_FILE_HELP = ("The project's moseq2-project.yaml, whose sessions and groups (in its config.yaml) the "
                      "command uses; by default the one in the folder the command runs in")


def _project_file():
    return project_file_option(_PROJECT_FILE_HELP)


def _sessions(ctx):
    """The sessions of the project the command runs on, as an index in memory; the command stops when the
    project has none to work on."""
    project = command_project(ctx, required=True)
    exclude = typed_exclude(ctx)
    index = session_index(project, exclude=exclude)
    if not index["files"]:
        if index.get("excluded") and exclude is not None:
            raise click.ClickException(
                f"Every session {project.bookkeeping_file} lists is in the list to leave out given with --exclude, "
                f"so `{ctx.info_name}` has no session to work on. Name fewer sessions with --exclude."
            )
        if index.get("excluded"):
            raise click.ClickException(
                f"Every session {project.bookkeeping_file} lists is under exclude in {project.config_file}, so "
                f"`{ctx.info_name}` has no session to work on. Remove at least one session from exclude in "
                f"{project.config_file}, or run with --no-exclude to leave none out for this run."
            )
        raise click.ClickException(
            f"{project.bookkeeping_file} lists no sessions, so `{ctx.info_name}` has none to work on. Run "
            f"`moseq2-extract aggregate-results` in {project.folder} (or the notebook's aggregation cell) to list "
            "the project's extractions."
        )
    return index


def _argument(name, missing, default):
    return project_argument(name, type=click.Path(exists=True, resolve_path=True), missing=missing, default=default)


def _model(what, name="model-fit"):
    """A command's fitted-model argument, ``what`` saying what the command does with it; the project's current
    model when left out."""
    return _argument(name, f"the fitted model {what}, the .p file learn-model writes (models/model.p in a project, "
                           "or the model get-best-model picked).", PROJECT_MODEL)


def _output_file(name, help=None):
    """A plot's ``--output-file``; without it, the figure is ``plots/<name>`` in the project (``_plot_file``)."""
    return click.option("--output-file", type=click.Path(), default=None, show_default=False,
                        help=help or f"Path of the figure, without its extension (default: plots/{name} in the "
                                     "project folder)")


def _plot_file(ctx, output_file, name):
    """``--output-file``, or ``plots/<name>`` in the command's project, decided when the command runs."""
    return output_file or str(output_folder(ctx) / "plots" / name)


@cli.command(
    name="add-group",
    help="Set the group of the project's sessions whose metadata matches, in the project's config.yaml (groups)",
)
@click_options(AddGroupOptions,
               order=("key", "value", "group", "exact", "lowercase", "negative", "project_file"),
               others={"project_file": _project_file()})
@click.pass_context
def add_group(ctx, **_):
    project = command_project(ctx, required=True)
    steps.add_group(project, AddGroupOptions.resolve(explicit=explicit_values(ctx, AddGroupOptions)))


@cli.command(
    name="get-best-model",
    help="Get the model closest to model-free Changepoints, given the objective",
)
@click_options(BestModelOptions,
               order=("model_dir", "cp_path", "output_file", "plot_all", "ext", "objective", "project_file"),
               others={"model_dir": _argument("model-dir",
                                              "the folder holding the models to compare, the OUTPUT_DIR given to "
                                              "moseq2-model kappa-scan (models in the project folder, where the "
                                              "notebook writes them).",
                                              in_project("models", make="Run `moseq2-model kappa-scan` in the "
                                                                        "project folder to fit models there.")),
                       "cp_path": _argument("cp-path", "the changepoints file moseq2-pca compute-changepoints "
                                                       "writes (_pca/changepoints.h5 in a project).",
                                            PROJECT_CHANGEPOINTS),
                       "output_file": project_argument(
                           "output-file", type=click.Path(exists=False, resolve_path=True),
                           default=in_project("plots", "model_vs_pc_changepoints"),
                           missing="the path of the comparison figure to write, without its extension (the "
                                   "notebook writes plots/model_vs_pc_changepoints in the project folder)."),
                       "project_file": project_file_option(
                           "The project's moseq2-project.yaml, which lists the models compared and the one picked "
                           "as its current model; by default the project in the folder the command runs in")})
@click.pass_context
def get_best_fit_model(ctx, model_dir, cp_path, output_file, **_):
    steps.get_best_model(model_dir, cp_path, output_file, _resolve(ctx, BestModelOptions),
                         project=command_project(ctx))


@cli.command(
    name="make-df",
    help="Make a dataframe containing syllable and scalar information from the project's sessions and a model fit file",
)
# no settings: the output file's extension picks the format
@_model("whose syllables to tabulate")
@click.option("--output-file", type=click.Path(), default=None, show_default=False,
              help="Filename to store dataframe (default: moseq_df.csv in the project folder)")
@_project_file()
@exclude_options()["exclude"]
@exclude_options()["no_exclude"]
@click.pass_context
def make_df(ctx, model_fit, output_file, **_):
    sessions = _sessions(ctx)
    output_file = output_file or str(output_folder(ctx) / "moseq_df.csv")
    steps.make_df(model_fit, sessions, output_file)
    print(f"Dataframe successfully generated at:\n    {output_file}")


@cli.command(
    name="make-crowd-movies",
    help="write movies of overlaid examples of the rodent perform a given syllable",
)
@click_options(CrowdMovieOptions, order=(
    "model_path", "max_syllable", "n_explained", "max_examples", "processes", "separate_by",
    "specific_syllable", "session_names", "sort", "count", "output_dir", "gaussfilter_space", "medfilter_space",
    "min_height", "max_height", "raw_size", "scale", "cmap", "max_dur", "min_dur", "legacy_jitter_fix",
    "frame_path", "progress_bar", "pad", "seed", "select_median_duration_instances", "project_file", "exclude",
    "no_exclude",
), others={
    "project_file": _project_file(),
    **exclude_options(),
    "model_path": _model("whose syllables to render as crowd movies", "model-path"),
    "output_dir": click.option("--output-dir", "-o", type=click.Path(), default=None, show_default=False,
                               help="Path to store files (default: crowd_movies in the project folder)"),
})
@click.pass_context
def make_crowd_movies(ctx, model_path, output_dir, **_):
    sessions = _sessions(ctx)
    output_dir = output_dir or str(output_folder(ctx) / "crowd_movies")
    steps.make_crowd_movies(sessions, model_path, output_dir, _resolve(ctx, CrowdMovieOptions))
    print(f"Crowd movies successfully generated in {output_dir}.")


@cli.command(
    name="plot-scalar-summary",
    help="Plot a scalar summary scatter plot for each group.",
)
@click_options(ScalarSummaryOptions, order=("output_file", "colors", "project_file", "exclude", "no_exclude"), others={
    "project_file": _project_file(),
    "output_file": _output_file("scalars", "Path of the figure, without its extension and the _summary added to its "
                                           "name (default: plots/scalars_summary in the project folder)"),
    **exclude_options(),
})
@click.pass_context
def plot_scalar_summary(ctx, output_file, **_):
    steps.plot_scalar_summary(_sessions(ctx), _plot_file(ctx, output_file, "scalars"),
                              _resolve(ctx, ScalarSummaryOptions))
    print("Sucessfully plotted scalar summary")


@cli.command(
    name="plot-group-position-heatmaps", help="Plots position heatmaps for each group"
)
@click_options(HeatmapOptions, order=("output_file", "normalize", "project_file", "exclude", "no_exclude"), others={
    "project_file": _project_file(),
    "output_file": _output_file("group_heat_map"),
    **exclude_options(),
})
@click.pass_context
def plot_group_position_heatmaps(ctx, output_file, **_):
    steps.plot_group_position_heatmaps(_sessions(ctx), _plot_file(ctx, output_file, "group_heat_map"),
                                       _resolve(ctx, HeatmapOptions))
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-verbose-position-heatmaps",
    help="Plots a position heatmap for each session in the project.",
)
@click_options(HeatmapOptions, order=("output_file", "normalize", "project_file", "exclude", "no_exclude"), others={
    "project_file": _project_file(),
    "output_file": _output_file("session_heat_map"),
    **exclude_options(),
})
@click.pass_context
def plot_verbose_position_heatmaps(ctx, output_file, **_):
    steps.plot_verbose_position_heatmaps(_sessions(ctx), _plot_file(ctx, output_file, "session_heat_map"),
                                         _resolve(ctx, HeatmapOptions))
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-transition-graph",
    help="Plot the transition graph depicting the transition probabilities between syllables.",
)
@click_options(TransitionGraphOptions, order=(
    "model_fit", "max_syllable", "n_explained", "group", "output_file", "normalize",
    "edge_threshold", "usage_threshold", "layout", "keep_orphans", "orphan_weight", "arrows", "sort", "count",
    "scale_node_by_usage", "width_per_group", "project_file", "exclude", "no_exclude",
), others={
    "project_file": _project_file(),
    **exclude_options(),
    "model_fit": _model("whose transitions to plot"),
    "output_file": _output_file("transitions"),
})
@click.pass_context
def plot_transition_graph(ctx, model_fit, output_file, **_):
    steps.plot_transition_graph(_sessions(ctx), model_fit, _plot_file(ctx, output_file, "transitions"),
                                _resolve(ctx, TransitionGraphOptions))


@cli.command(
    name="plot-stats",
    help="Plot syllable usages with different sorting,coloring and grouping capabilities",
)
@click_options(StatsPlotOptions, order=(
    "model_fit", "stat", "output_file", "sort", "figsize", "count", "max_syllable", "n_explained",
    "group", "ordering", "ctrl_group", "exp_group", "colors", "project_file", "exclude", "no_exclude",
), others={
    "project_file": _project_file(),
    **exclude_options(),
    "model_fit": _model("whose syllable statistics to plot"),
    "output_file": _output_file("syll_stat"),
})
@click.pass_context
def plot_stats(ctx, model_fit, output_file, **_):
    opts = _resolve(ctx, StatsPlotOptions)
    steps.plot_stats(_sessions(ctx), model_fit, _plot_file(ctx, output_file, "syll_stat"), opts)
    click.echo(f'Syllable {opts.stat} graph created successfully')


cli.add_command(print_record_command())

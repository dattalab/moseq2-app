"""
CLI for visualizing the results.

Each pipeline command's settings are generated from its options class in
``moseq2_viz.options``; the command resolves the settings and calls its step in
``moseq2_viz.steps``. ``make-crowd-movies`` reads the ``crowd_movies`` section
of the ``config.yaml`` in the folder it runs in.

The commands work on a project's sessions: those its ``moseq2-project.yaml``
lists, each with its group from the project's ``config.yaml``
(``--project-file``, by default the one in the folder the command runs in).
Without one they stop, saying where they looked.
"""

import os
from pathlib import Path

import click

from moseq2_common.bookkeeping import session_index
from moseq2_common.options import (
    click_options,
    command_project,
    command_settings,
    explicit_values,
    project_file_option,
)
from moseq2_common.project import find_project
from moseq2_viz import steps
from moseq2_viz.helpers.wrappers import copy_h5_metadata_to_yaml_wrapper
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


@click.group(context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _resolve(ctx, cls):
    """``cls`` resolved from the typed options, and the config.yaml of the project the command runs in."""
    config = command_settings(None, cls, stage_classes=CROWD_MOVIE_STAGE if cls is CrowdMovieOptions else (cls,))
    return cls.resolve(config=config, explicit=explicit_values(ctx, cls))


_PROJECT_FILE_HELP = ("The project's moseq2-project.yaml, whose sessions and groups (in its config.yaml) the "
                      "command uses; by default the one in the folder the command runs in")


def _project_file():
    return project_file_option(_PROJECT_FILE_HELP)


def _sessions(ctx, project_file):
    """The sessions of the project the command runs on, as an index in memory."""
    return session_index(command_project(project_file, f"moseq2-viz {ctx.info_name}", required=True))


def _argument(name):
    return click.argument(name, type=click.Path(exists=True, resolve_path=True))


def _output_file(default, help=None):
    return click.option("--output-file", type=click.Path(), default=default, help=help)


@cli.command(
    name="add-group",
    help="Set the group of the project's sessions whose metadata matches, in the project's config.yaml (groups)",
)
@click_options(AddGroupOptions,
               order=("key", "value", "group", "exact", "lowercase", "negative", "project_file"),
               others={"project_file": _project_file()})
@click.pass_context
def add_group(ctx, project_file, **_):
    project = command_project(project_file, "moseq2-viz add-group", required=True)
    steps.add_group(project, AddGroupOptions.resolve(explicit=explicit_values(ctx, AddGroupOptions)))


@cli.command(
    name="get-best-model",
    help="Get the model closest to model-free Changepoints, given the objective",
)
@click_options(BestModelOptions,
               order=("model_dir", "cp_path", "output_file", "plot_all", "ext", "fps", "objective"),
               others={"model_dir": _argument("model-dir"), "cp_path": _argument("cp-path"),
                       "output_file": click.argument("output-file",
                                                     type=click.Path(exists=False, resolve_path=True))})
@click.pass_context
def get_best_fit_model(ctx, model_dir, cp_path, output_file, **_):
    steps.get_best_model(model_dir, cp_path, output_file, _resolve(ctx, BestModelOptions), project=find_project())


# recurse through directories, find h5 files with completed extractions, make a manifest
# and copy the contents to a new directory
@cli.command(
    name="copy-h5-metadata-to-yaml",
    help="Copy metadata within an h5 file to a yaml file.",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
def copy_h5_metadata_to_yaml(input_dir):
    copy_h5_metadata_to_yaml_wrapper(input_dir)


@cli.command(
    name="make-df",
    help="Make a dataframe containing syllable and scalar information from the project's sessions and a model fit file",
)
# no settings: the output file's extension picks the format
@_argument("model-fit")
@_output_file(Path.cwd() / "moseq_df.csv", "Filename to store dataframe")
@_project_file()
@click.pass_context
def make_df(ctx, model_fit, output_file, project_file):
    steps.make_df(model_fit, _sessions(ctx, project_file), output_file)
    print(f"Dataframe successfully generated at:\n    {Path(output_file)}")


@cli.command(
    name="make-crowd-movies",
    help="write movies of overlaid examples of the rodent perform a given syllable",
)
@click_options(CrowdMovieOptions, order=(
    "model_path", "max_syllable", "n_explained", "max_examples", "processes", "separate_by",
    "specific_syllable", "session_names", "sort", "count", "output_dir", "gaussfilter_space", "medfilter_space",
    "min_height", "max_height", "raw_size", "scale", "cmap", "max_dur", "min_dur", "legacy_jitter_fix",
    "frame_path", "progress_bar", "pad", "seed", "select_median_duration_instances", "project_file",
), others={
    "project_file": _project_file(),
    "model_path": _argument("model-path"),
    "output_dir": click.option("--output-dir", "-o", type=click.Path(),
                               default=os.path.join(os.getcwd(), "crowd_movies"), help="Path to store files"),
})
@click.pass_context
def make_crowd_movies(ctx, project_file, model_path, output_dir, **_):
    steps.make_crowd_movies(_sessions(ctx, project_file), model_path, output_dir, _resolve(ctx, CrowdMovieOptions))
    print(f"Crowd movies successfully generated in {output_dir}.")


@cli.command(
    name="plot-scalar-summary",
    help="Plot a scalar summary scatter plot for each group.",
)
@click_options(ScalarSummaryOptions, order=("output_file", "colors", "project_file"), others={
    "project_file": _project_file(),
    "output_file": _output_file(os.path.join(os.getcwd(), "scalars")),
})
@click.pass_context
def plot_scalar_summary(ctx, project_file, output_file, **_):
    steps.plot_scalar_summary(_sessions(ctx, project_file), output_file, _resolve(ctx, ScalarSummaryOptions))
    print("Sucessfully plotted scalar summary")


@cli.command(
    name="plot-group-position-heatmaps", help="Plots position heatmaps for each group"
)
@click_options(HeatmapOptions, order=("output_file", "normalize", "project_file"), others={
    "project_file": _project_file(),
    "output_file": _output_file(os.path.join(os.getcwd(), "group_heat_map")),
})
@click.pass_context
def plot_group_position_heatmaps(ctx, project_file, output_file, **_):
    steps.plot_group_position_heatmaps(_sessions(ctx, project_file), output_file, _resolve(ctx, HeatmapOptions))
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-verbose-position-heatmaps",
    help="Plots a position heatmap for each session in the project.",
)
@click_options(HeatmapOptions, order=("output_file", "normalize", "project_file"), others={
    "project_file": _project_file(),
    "output_file": _output_file(os.path.join(os.getcwd(), "session_heat_map")),
})
@click.pass_context
def plot_verbose_position_heatmaps(ctx, project_file, output_file, **_):
    steps.plot_verbose_position_heatmaps(_sessions(ctx, project_file), output_file, _resolve(ctx, HeatmapOptions))
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-transition-graph",
    help="Plot the transition graph depicting the transition probabilities between syllables.",
)
@click_options(TransitionGraphOptions, order=(
    "model_fit", "max_syllable", "n_explained", "group", "output_file", "normalize",
    "edge_threshold", "usage_threshold", "layout", "keep_orphans", "orphan_weight", "arrows", "sort", "count",
    "scale_node_by_usage", "width_per_group", "project_file",
), others={
    "project_file": _project_file(),
    "model_fit": _argument("model-fit"),
    "output_file": _output_file(os.path.join(os.getcwd(), "transitions"), "Filename to store plot"),
})
@click.pass_context
def plot_transition_graph(ctx, project_file, model_fit, output_file, **_):
    steps.plot_transition_graph(_sessions(ctx, project_file), model_fit, output_file,
                                _resolve(ctx, TransitionGraphOptions))


@cli.command(
    name="plot-stats",
    help="Plot syllable usages with different sorting,coloring and grouping capabilities",
)
@click_options(StatsPlotOptions, order=(
    "model_fit", "stat", "output_file", "sort", "figsize", "count", "max_syllable", "n_explained",
    "group", "ordering", "ctrl_group", "exp_group", "colors", "project_file",
), others={
    "project_file": _project_file(),
    "model_fit": _argument("model-fit"),
    "output_file": _output_file(os.path.join(os.getcwd(), "syll_stat"), "Filename to store plot"),
})
@click.pass_context
def plot_stats(ctx, project_file, model_fit, output_file, **_):
    opts = _resolve(ctx, StatsPlotOptions)
    steps.plot_stats(_sessions(ctx, project_file), model_fit, output_file, opts)
    click.echo(f'Syllable {opts.stat} graph created successfully')

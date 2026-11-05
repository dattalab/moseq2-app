"""
CLI for visualizing the results.

Each pipeline command's settings are generated from its options class in
``moseq2_viz.options``; the command resolves the settings and calls its step in
``moseq2_viz.steps``. ``make-crowd-movies`` reads the ``crowd_movies`` section
of the ``config.yaml`` in the folder it runs in.
"""

import os
from pathlib import Path

import click

from moseq2_common.options import click_options, command_settings, explicit_values
from moseq2_common.project import find_project, standalone_note
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


def _argument(name):
    return click.argument(name, type=click.Path(exists=True, resolve_path=True))


def _output_file(default, help=None):
    return click.option("--output-file", type=click.Path(), default=default, help=help)


@cli.command(
    name="add-group",
    help="Change group name for the sessions in index file (moseq2-index.yaml)",
)
@click_options(AddGroupOptions,
               order=("index_file", "key", "value", "group", "exact", "lowercase", "negative"),
               others={"index_file": _argument("index-file")})
@click.pass_context
def add_group(ctx, index_file, **_):
    steps.add_group(index_file, AddGroupOptions.resolve(explicit=explicit_values(ctx, AddGroupOptions)))


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
    steps.get_best_model(model_dir, cp_path, output_file, _resolve(ctx, BestModelOptions))


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
    help="Make a dataframe containing syllable and scalar information from an index and model fit file",
)
# no settings: the output file's extension picks the format
@_argument("model-fit")
@_argument("index-file")
@_output_file(Path.cwd() / "moseq_df.csv", "Filename to store dataframe")
def make_df(model_fit, index_file, output_file):
    if find_project() is None:
        click.echo(standalone_note(os.getcwd()))
    steps.make_df(model_fit, index_file, output_file)
    print(f"Dataframe successfully generated at:\n    {Path(output_file)}")


@cli.command(
    name="make-crowd-movies",
    help="write movies of overlaid examples of the rodent perform a given syllable",
)
@click_options(CrowdMovieOptions, order=(
    "index_file", "model_path", "max_syllable", "n_explained", "max_examples", "processes", "separate_by",
    "specific_syllable", "session_names", "sort", "count", "output_dir", "gaussfilter_space", "medfilter_space",
    "min_height", "max_height", "raw_size", "scale", "cmap", "max_dur", "min_dur", "legacy_jitter_fix",
    "frame_path", "progress_bar", "pad", "seed", "select_median_duration_instances",
), others={
    "index_file": _argument("index-file"),
    "model_path": _argument("model-path"),
    "output_dir": click.option("--output-dir", "-o", type=click.Path(),
                               default=os.path.join(os.getcwd(), "crowd_movies"), help="Path to store files"),
})
@click.pass_context
def make_crowd_movies(ctx, index_file, model_path, output_dir, **_):
    steps.make_crowd_movies(index_file, model_path, output_dir, _resolve(ctx, CrowdMovieOptions))
    print(f"Crowd movies successfully generated in {output_dir}.")


@cli.command(
    name="plot-scalar-summary",
    help="Plot a scalar summary scatter plot for each group.",
)
@click_options(ScalarSummaryOptions, order=("index_file", "output_file", "colors"), others={
    "index_file": _argument("index-file"),
    "output_file": _output_file(os.path.join(os.getcwd(), "scalars")),
})
@click.pass_context
def plot_scalar_summary(ctx, index_file, output_file, **_):
    steps.plot_scalar_summary(index_file, output_file, _resolve(ctx, ScalarSummaryOptions))
    print("Sucessfully plotted scalar summary")


@cli.command(
    name="plot-group-position-heatmaps", help="Plots position heatmaps for each group"
)
@click_options(HeatmapOptions, order=("index_file", "output_file", "normalize"), others={
    "index_file": _argument("index-file"),
    "output_file": _output_file(os.path.join(os.getcwd(), "group_heat_map")),
})
@click.pass_context
def plot_group_position_heatmaps(ctx, index_file, output_file, **_):
    steps.plot_group_position_heatmaps(index_file, output_file, _resolve(ctx, HeatmapOptions))
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-verbose-position-heatmaps",
    help="Plots a position heatmap for each session in the index file.",
)
@click_options(HeatmapOptions, order=("index_file", "output_file", "normalize"), others={
    "index_file": _argument("index-file"),
    "output_file": _output_file(os.path.join(os.getcwd(), "session_heat_map")),
})
@click.pass_context
def plot_verbose_position_heatmaps(ctx, index_file, output_file, **_):
    steps.plot_verbose_position_heatmaps(index_file, output_file, _resolve(ctx, HeatmapOptions))
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-transition-graph",
    help="Plot the transition graph depicting the transition probabilities between syllables.",
)
@click_options(TransitionGraphOptions, order=(
    "index_file", "model_fit", "max_syllable", "n_explained", "group", "output_file", "normalize",
    "edge_threshold", "usage_threshold", "layout", "keep_orphans", "orphan_weight", "arrows", "sort", "count",
    "scale_node_by_usage", "width_per_group",
), others={
    "index_file": _argument("index-file"),
    "model_fit": _argument("model-fit"),
    "output_file": _output_file(os.path.join(os.getcwd(), "transitions"), "Filename to store plot"),
})
@click.pass_context
def plot_transition_graph(ctx, index_file, model_fit, output_file, **_):
    steps.plot_transition_graph(index_file, model_fit, output_file,
                                _resolve(ctx, TransitionGraphOptions))


@cli.command(
    name="plot-stats",
    help="Plot syllable usages with different sorting,coloring and grouping capabilities",
)
@click_options(StatsPlotOptions, order=(
    "index_file", "model_fit", "stat", "output_file", "sort", "figsize", "count", "max_syllable", "n_explained",
    "group", "ordering", "ctrl_group", "exp_group", "colors",
), others={
    "index_file": _argument("index-file"),
    "model_fit": _argument("model-fit"),
    "output_file": _output_file(os.path.join(os.getcwd(), "syll_stat"), "Filename to store plot"),
})
@click.pass_context
def plot_stats(ctx, index_file, model_fit, output_file, **_):
    opts = _resolve(ctx, StatsPlotOptions)
    steps.plot_stats(index_file, model_fit, output_file, opts)
    click.echo(f'Syllable {opts.stat} graph created successfully')

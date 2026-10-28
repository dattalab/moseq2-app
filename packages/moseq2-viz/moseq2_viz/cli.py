"""
CLI for visualizing the results.
"""

import os
import click
from pathlib import Path
from moseq2_viz.helpers.wrappers import (
    add_group_wrapper,
    plot_syllable_stat_wrapper,
    plot_scalar_summary_wrapper,
    plot_transition_graph_wrapper,
    copy_h5_metadata_to_yaml_wrapper,
    make_crowd_movies_wrapper,
    plot_verbose_pdfs_wrapper,
    plot_mean_group_position_pdf_wrapper,
    get_best_fit_model_wrapper,
    make_df_wrapper,
)

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass


@cli.command(
    name="add-group",
    help="Change group name for the sessions in index file (moseq2-index.yaml)",
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--key",
    "-k",
    type=str,
    default="SubjectName",
    help="metadata field to search for value to match",
)
@click.option(
    "--value", "-v", type=str, default="Mouse", help="Value to match for", multiple=True
)
@click.option("--group", "-g", type=str, default="Group1", help="Group name to set")
@click.option("--exact", "-e", type=bool, is_flag=True, help="Exact match only")
@click.option(
    "--lowercase",
    type=bool,
    is_flag=True,
    help="Set metadata info to lower case for matching",
)
@click.option(
    "-n",
    "--negative",
    type=bool,
    is_flag=True,
    help="Negative match (everything that does not match is included)",
)
def add_group(index_file, **config_data):
    add_group_wrapper(index_file, config_data)


@cli.command(
    name="get-best-model",
    help="Get the model closest to model-free Changepoints, given the objective",
)
@click.argument("model-dir", type=click.Path(exists=True, resolve_path=True))
@click.argument("cp-path", type=click.Path(exists=True, resolve_path=True))
@click.argument("output-file", type=click.Path(exists=False, resolve_path=True))
@click.option("--plot-all", is_flag=True, help="Plot all included model results")
@click.option(
    "--ext",
    type=str,
    default="p",
    help="Model extensions to look for in input directory",
)
@click.option("--fps", type=int, default=30, help="Frames per second")
@click.option(
    "--objective",
    type=str,
    default="duration (mean match)",
    help="The objective finds the best model when comparing with model-free changepoints",
)
def get_best_fit_model(model_dir, cp_path, output_file, plot_all, ext, fps, objective):
    get_best_fit_model_wrapper(
        model_dir, cp_path, output_file, plot_all, ext, fps, objective
    )


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
@click.argument("model-fit", type=click.Path(exists=True, resolve_path=True))
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--output-file",
    type=click.Path(),
    default=Path.cwd() / "moseq_df.csv",
    help="Filename to store dataframe",
)
def make_df(model_fit, index_file, output_file):
    if isinstance(output_file, (str, click.Path)):
        output_file = Path(output_file)
    extension = output_file.suffix[1:]
    if extension not in ["csv", "parquet"]:
        raise ValueError(f"Extension {extension} is not supported. Please use .csv or .parquet.")
    make_df_wrapper(model_fit, index_file, output_file, extension)
    print(f"Dataframe successfully generated at:\n    {output_file}")


@cli.command(
    name="make-crowd-movies",
    help="write movies of overlaid examples of the rodent perform a given syllable",
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.argument("model-path", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--max-syllable", type=int, default=None,
    help="Max syllable to include (default: auto, enough to cover --n-explained % of usage)"
)
@click.option(
    "--n-explained", type=int, default=99,
    help="Percent of usage to cover when --max-syllable is not set (default: 99)"
)
@click.option(
    "--max-examples", "-m", type=int, default=40, help="number of examples to show"
)
@click.option(
    "--processes",
    type=int,
    default=None,
    help="number of processes to use for rendering crowd movies. Default None uses every process",
)
@click.option(
    "--separate-by",
    type=click.Choice(["default", "groups", "sessions", "subjects"]),
    default="default",
    help="generate crowd movies by specified grouping",
)
@click.option(
    "--specific-syllable",
    type=int,
    default=None,
    help="index of the specific syllable to render",
)
@click.option(
    "--session-names",
    "-s",
    default=[],
    type=str,
    help="specific sessions to create crowd movies from",
    multiple=True,
)
@click.option("--sort", type=bool, default=True, help="sort syllables by usage")
@click.option(
    "--count",
    type=click.Choice(["usage", "frames"]),
    default="usage",
    help="How to quantify syllable usage",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "crowd_movies"),
    help="Path to store files",
)
@click.option(
    "--gaussfilter-space",
    default=(0, 0),
    type=(float, float),
    help="x sigma and y sigma for Gaussian spatial filter to apply to data",
)
@click.option(
    "--medfilter-space",
    default=[0],
    type=int,
    help="kernel size for median spatial filter",
    multiple=True,
)
@click.option(
    "--min-height", type=int, default=5, help="Minimum height for scaling videos"
)
@click.option(
    "--max-height", type=int, default=80, help="Maximum height for scaling videos"
)
@click.option(
    "--raw-size",
    type=(int, int),
    default=(512, 424),
    help="frame dimension of original videos",
)
@click.option("--scale", type=float, default=1, help="Scaling from pixel units to mm")
@click.option(
    "--cmap",
    type=str,
    default="jet",
    help="Name of valid Matplotlib colormap for false-coloring images",
)
@click.option(
    "--max-dur",
    default=60,
    help="Exclude syllables longer than this number of frames (None for no limit)",
)
@click.option(
    "--min-dur", default=0, help="Exclude syllables shorter than this number of frames"
)
@click.option(
    "--legacy-jitter-fix",
    default=False,
    type=bool,
    help="Set to true if you notice jitter in your crowd movies",
)
@click.option(
    "--frame-path", default="frames", type=str, help="Path to depth frames in h5 file"
)
@click.option("--progress-bar", "-p", is_flag=True, help="Show verbose progress bars.")
@click.option("--pad", default=30, help="Pad crowd movie videos with this many frames.")
@click.option(
    "--seed",
    default=0,
    type=int,
    help="Defines random seed for selecting syllable instances to plot",
)
def make_crowd_movies(index_file, model_path, output_dir, **config_data):
    make_crowd_movies_wrapper(index_file, model_path, output_dir, config_data)
    print(f"Crowd movies successfully generated in {output_dir}.")


@cli.command(
    name="plot-scalar-summary",
    help="Plot a scalar summary scatter plot for each group.",
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--output-file", type=click.Path(), default=os.path.join(os.getcwd(), "scalars")
)
@click.option(
    "-c",
    "--colors",
    type=str,
    default=None,
    help="Colors to plot groups with.",
    multiple=True,
)
def plot_scalar_summary(index_file, output_file, colors):
    plot_scalar_summary_wrapper(index_file, output_file, colors=colors)
    print("Sucessfully plotted scalar summary")


@cli.command(
    name="plot-group-position-heatmaps", help="Plots position heatmaps for each group"
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--output-file",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "group_heat_map"),
)
@click.option(
    "--normalize",
    type=bool,
    is_flag=True,
    help="normalize the PDF so that min and max values range from 0-1",
)
def plot_group_position_heatmaps(index_file, output_file, normalize):
    plot_mean_group_position_pdf_wrapper(index_file, output_file, normalize=normalize)
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-verbose-position-heatmaps",
    help="Plots a position heatmap for each session in the index file.",
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--output-file",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "session_heat_map"),
)
@click.option(
    "--normalize",
    type=bool,
    is_flag=True,
    help="normalize the PDF so that min and max values range from 0-1",
)
def plot_verbose_position_heatmaps(index_file, output_file, normalize):
    plot_verbose_pdfs_wrapper(index_file, output_file, normalize=normalize)
    print("Sucessfully plotted mean group heatmaps")


@cli.command(
    name="plot-transition-graph",
    help="Plot the transition graph depicting the transition probabilities between syllables.",
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.argument("model-fit", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--max-syllable", type=int, default=None,
    help="Max syllable to include (default: auto, enough to cover --n-explained % of usage)"
)
@click.option(
    "--n-explained", type=int, default=99,
    help="Percent of usage to cover when --max-syllable is not set (default: 99)"
)
@click.option(
    "-g",
    "--group",
    type=str,
    default=None,
    help="the groups to include in the transition graph",
    multiple=True,
)
@click.option(
    "--output-file",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "transitions"),
    help="Filename to store plot",
)
@click.option(
    "--normalize",
    type=click.Choice(["bigram", "rows", "columns"]),
    default="bigram",
    help="How to normalize transition probabilities",
)
@click.option(
    "--edge-threshold", type=float, default=0.001, help="Threshold for edges to show"
)
@click.option(
    "--usage-threshold", type=float, default=0, help="Threshold for nodes to show"
)
@click.option("--layout", type=str, default="spring", help="networkx layout algorithm")
@click.option(
    "--keep-orphans", "-k", type=bool, is_flag=True, help="Show orphaned nodes"
)
@click.option(
    "--orphan-weight", type=float, default=0, help="Weight for non-existent connections"
)
@click.option("--arrows", type=bool, is_flag=True, help="Show arrows")
@click.option("--sort", type=bool, default=True, help="Sort syllables by usage")
@click.option(
    "--count",
    type=click.Choice(["usage", "frames"]),
    default="usage",
    help="How to quantify syllable usage",
)
@click.option(
    "--edge-scaling",
    type=float,
    default=250,
    help="Scale factor from transition probabilities to edge width",
)
@click.option(
    "--node-scaling", type=float, default=1e5, help="Scale factor for nodes by usage"
)
@click.option(
    "--scale-node-by-usage",
    type=bool,
    default=True,
    help="Scale node sizes by usages probabilities",
)
@click.option(
    "--width-per-group", type=float, default=8, help="Width for figure canvas per group"
)
def plot_transition_graph(index_file, model_fit, output_file, **config_data):
    plot_transition_graph_wrapper(index_file, model_fit, output_file, config_data)


@cli.command(
    name="plot-stats",
    help="Plot syllable usages with different sorting,coloring and grouping capabilities",
)
@click.argument("index-file", type=click.Path(exists=True, resolve_path=True))
@click.argument("model-fit", type=click.Path(exists=True, resolve_path=True))
@click.option("--stat", type=str, default="usage", help="Statistic to plot")
@click.option(
    "--output-file",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "syll_stat"),
    help="Filename to store plot",
)
@click.option("--sort", type=bool, default=True, help="Sort syllables by usage")
@click.option(
    "--figsize", type=tuple, default=(10, 5), help="Figure size of the plotted figure."
)
@click.option(
    "--count",
    type=click.Choice(["usage", "frames"]),
    default="usage",
    help="How to relabel syllables",
)
@click.option(
    "--max-syllable", type=int, default=None,
    help="Max syllable to include (default: auto, enough to cover --n-explained % of usage)"
)
@click.option(
    "--n-explained", type=int, default=99,
    help="Percent of usage to cover when --max-syllable is not set (default: 99)"
)
@click.option(
    "-g",
    "--group",
    type=str,
    default=None,
    help="Name of group(s) to show",
    multiple=True,
)
@click.option(
    "-o", "--ordering", type=str, default="stat", help="How to order syllables in plot"
)
@click.option("--ctrl-group", type=str, default=None, help="Name of control group")
@click.option("--exp-group", type=str, default=None, help="Name of experimental group")
@click.option(
    "-c",
    "--colors",
    type=str,
    default=None,
    help="Colors to plot groups with",
    multiple=True,
)
def plot_stats(index_file, model_fit, output_file, **cli_kwargs):
    plot_syllable_stat_wrapper(model_fit, index_file, output_file, **cli_kwargs)
    click.echo(f'Syllable {cli_kwargs["stat"]} graph created successfully')

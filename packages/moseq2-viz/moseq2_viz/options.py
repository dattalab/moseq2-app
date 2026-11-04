"""The settings of each analysis step, declared once.

One class per command. Each field carries its default, its command-line flags,
its click type and its help text; ``cli.py`` generates the commands' options
from these classes, and ``steps.py`` takes an instance. The files a step works
on (the commands' arguments, ``--output-file`` and ``--output-dir``) are not
settings: they are the steps' arguments and plain options in ``cli.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import click

from moseq2_common.options import Options, option


@dataclass(frozen=True, kw_only=True)
class AddGroupOptions(Options):
    """``add-group``."""

    key: str = option("SubjectName", "--key", "-k", type=str,
                      help="metadata field to search for value to match")
    value: tuple | str = option("Mouse", "--value", "-v", type=str, multiple=True, help="Value to match for")
    group: str = option("Group1", "--group", "-g", type=str, help="Group name to set")
    exact: bool = option(False, "--exact", "-e", type=bool, is_flag=True, help="Exact match only")
    lowercase: bool = option(False, "--lowercase", type=bool, is_flag=True,
                             help="Set metadata info to lower case for matching")
    negative: bool = option(False, "-n", "--negative", type=bool, is_flag=True,
                            help="Negative match (everything that does not match is included)")


@dataclass(frozen=True, kw_only=True)
class BestModelOptions(Options):
    """``get-best-model``."""

    plot_all: bool = option(False, "--plot-all", is_flag=True, help="Plot all included model results")
    ext: str = option("p", "--ext", type=str, help="Model extensions to look for in input directory")
    fps: int = option(30, "--fps", type=int, help="Frames per second")
    objective: str = option("duration (mean match)", "--objective", type=str,
                            help="The objective finds the best model when comparing with model-free changepoints")


@dataclass(frozen=True, kw_only=True)
class SyllableCoverageOptions(Options):
    """How many syllables a step includes."""

    max_syllable: int | None = option(
        None, "--max-syllable", type=int,
        help="Max syllable to include (default: auto, enough to cover --n-explained % of usage)")
    n_explained: int = option(99, "--n-explained", type=int,
                              help="Percent of usage to cover when --max-syllable is not set (default: 99)")


def available_cpus() -> int:
    """The CPUs this process may run on.

    Its CPU affinity where the operating system reports one (Linux, which is how a
    SLURM allocation smaller than the node shows), else the machine's count (macOS,
    Windows).
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass(frozen=True, kw_only=True)
class CrowdMovieOptions(SyllableCoverageOptions):
    """``make-crowd-movies``, the syllable labeler and the crowd-movie comparison tool.

    ``processes`` left as None becomes the number of CPUs available to the process
    (``available_cpus``).
    """

    max_examples: int = option(20, "--max-examples", "-m", type=int, help="number of examples to show")
    processes: int | None = option(
        None, "--processes", type=int,
        help="number of processes to use for rendering crowd movies (default: every CPU "
             "available to this process)")
    separate_by: str = option("default", "--separate-by",
                              type=click.Choice(["default", "groups", "sessions", "subjects"]),
                              help="generate crowd movies by specified grouping")
    specific_syllable: int | None = option(None, "--specific-syllable", type=int,
                                           help="index of the specific syllable to render")
    session_names: tuple = option((), "--session-names", "-s", type=str, multiple=True,
                                  help="specific sessions to create crowd movies from")
    sort: bool = option(True, "--sort", type=bool, help="sort syllables by usage")
    count: str = option("usage", "--count", type=click.Choice(["usage", "frames"]),
                        help="How to quantify syllable usage")
    gaussfilter_space: tuple = option((0, 0), "--gaussfilter-space", type=(float, float),
                                      help="x sigma and y sigma for Gaussian spatial filter to apply to data")
    medfilter_space: tuple = option((0,), "--medfilter-space", type=int, multiple=True,
                                    help="kernel size for median spatial filter")
    min_height: int = option(10, "--min-height", type=int, help="Minimum height for scaling videos")
    max_height: int = option(120, "--max-height", type=int, help="Maximum height for scaling videos")
    raw_size: tuple = option((512, 424), "--raw-size", type=(int, int), help="frame dimension of original videos")
    scale: float = option(1, "--scale", type=float, help="Scaling from pixel units to mm")
    cmap: str = option("jet", "--cmap", type=str,
                       help="Name of valid Matplotlib colormap for false-coloring images")
    max_dur: int = option(60, "--max-dur",
                          help="Exclude syllables longer than this number of frames (None for no limit)")
    min_dur: int = option(4, "--min-dur", help="Exclude syllables shorter than this number of frames")
    legacy_jitter_fix: bool = option(False, "--legacy-jitter-fix", type=bool,
                                     help="Set to true if you notice jitter in your crowd movies")
    frame_path: str = option("frames", "--frame-path", type=str, help="Path to depth frames in h5 file")
    progress_bar: bool = option(False, "--progress-bar", "-p", is_flag=True, help="Show verbose progress bars.")
    pad: int = option(10, "--pad", help="Pad crowd movie videos with this many frames.")
    seed: int = option(0, "--seed", type=int,
                       help="Defines random seed for selecting syllable instances to plot")

    def __post_init__(self):
        if self.processes is None:
            object.__setattr__(self, "processes", available_cpus())


@dataclass(frozen=True, kw_only=True)
class ScalarSummaryOptions(Options):
    """``plot-scalar-summary``."""

    colors: tuple | None = option(None, "-c", "--colors", type=str, multiple=True,
                                  help="Colors to plot groups with.")


@dataclass(frozen=True, kw_only=True)
class HeatmapOptions(Options):
    """``plot-group-position-heatmaps`` and ``plot-verbose-position-heatmaps``."""

    normalize: bool = option(False, "--normalize", type=bool, is_flag=True,
                             help="normalize the PDF so that min and max values range from 0-1")


@dataclass(frozen=True, kw_only=True)
class TransitionGraphOptions(SyllableCoverageOptions):
    """``plot-transition-graph``."""

    group: tuple | None = option(None, "-g", "--group", type=str, multiple=True,
                                 help="the groups to include in the transition graph")
    normalize: str = option("bigram", "--normalize", type=click.Choice(["bigram", "rows", "columns"]),
                            help="How to normalize transition probabilities")
    edge_threshold: float = option(0.001, "--edge-threshold", type=float, help="Threshold for edges to show")
    usage_threshold: float = option(0, "--usage-threshold", type=float, help="Threshold for nodes to show")
    layout: str = option("spring", "--layout", type=str, help="networkx layout algorithm")
    keep_orphans: bool = option(False, "--keep-orphans", "-k", type=bool, is_flag=True,
                                help="Show orphaned nodes")
    orphan_weight: float = option(0, "--orphan-weight", type=float, help="Weight for non-existent connections")
    arrows: bool = option(False, "--arrows", type=bool, is_flag=True, help="Show arrows")
    sort: bool = option(True, "--sort", type=bool, help="Sort syllables by usage")
    count: str = option("usage", "--count", type=click.Choice(["usage", "frames"]),
                        help="How to quantify syllable usage")
    scale_node_by_usage: bool = option(True, "--scale-node-by-usage", type=bool,
                                       help="Scale node sizes by usages probabilities")
    width_per_group: float = option(8, "--width-per-group", type=float, help="Width for figure canvas per group")


@dataclass(frozen=True, kw_only=True)
class StatsPlotOptions(SyllableCoverageOptions):
    """``plot-stats``."""

    stat: str = option("usage", "--stat", type=str, help="Statistic to plot")
    sort: bool = option(True, "--sort", type=bool, help="Sort syllables by usage")
    figsize: tuple = option((10, 5), "--figsize", type=tuple, help="Figure size of the plotted figure.")
    count: str = option("usage", "--count", type=click.Choice(["usage", "frames"]),
                        help="How to relabel syllables")
    group: tuple | None = option(None, "-g", "--group", type=str, multiple=True, help="Name of group(s) to show")
    ordering: str = option("stat", "-o", "--ordering", type=str, help="How to order syllables in plot")
    ctrl_group: str | None = option(None, "--ctrl-group", type=str, help="Name of control group")
    exp_group: str | None = option(None, "--exp-group", type=str, help="Name of experimental group")
    colors: tuple | None = option(None, "-c", "--colors", type=str, multiple=True,
                                  help="Colors to plot groups with")

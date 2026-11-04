"""The analysis pipeline steps, one function each.

The ``moseq2-viz`` commands (``cli.py``) and the notebook's model selection
(``gui.py``) resolve their settings into one options object and call these. A
step needs no ``config.yaml``: from a script, pass the paths and an options
object, such as ``plot_stats("moseq2-index.yaml", "model.p", "plots/usage",
StatsPlotOptions(stat="duration"))``.

Each step hands its work function (the ``*_wrapper`` in
``moseq2_viz.helpers.wrappers``) the options, as keywords or as the dict the
wrapper takes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moseq2_viz.helpers.wrappers import (
    add_group_wrapper,
    get_best_fit_model_wrapper,
    make_crowd_movies_wrapper,
    make_df_wrapper,
    plot_mean_group_position_pdf_wrapper,
    plot_scalar_summary_wrapper,
    plot_syllable_stat_wrapper,
    plot_transition_graph_wrapper,
    plot_verbose_pdfs_wrapper,
)
from moseq2_viz.options import (
    AddGroupOptions,
    BestModelOptions,
    CrowdMovieOptions,
    HeatmapOptions,
    ScalarSummaryOptions,
    StatsPlotOptions,
    TransitionGraphOptions,
)


@dataclass(frozen=True)
class AddGroupResult:
    #: The index file, rewritten with the new group names.
    index_file: Path


@dataclass(frozen=True)
class BestModelResult:
    #: The wrapper's summary: the best model for the objective, its kappa, and the other models' scores.
    best_model_info: dict
    #: The comparison figure, for the notebook to show.
    figure: Any
    #: The figure's ``.png`` and ``.pdf``; empty when no output file was named.
    figure_files: list[Path]


@dataclass(frozen=True)
class MakeDfResult:
    dataframe_file: Path
    #: The per-session syllable statistics written beside the dataframe.
    stats_file: Path


@dataclass(frozen=True)
class CrowdMoviesResult:
    #: The wrapper's returned paths, ``{"all": [...]}`` or one list per group or session.
    movies: dict


@dataclass(frozen=True)
class FiguresResult:
    #: The figure's ``.png`` and ``.pdf``.
    figure_files: list[Path]


def _figures(output_file, suffix=""):
    """The files ``moseq2_viz.viz.save_fig`` writes for ``output_file``."""
    return [Path(f"{output_file}{suffix}.png"), Path(f"{output_file}{suffix}.pdf")]


def add_group(index_file: str, opts: AddGroupOptions) -> AddGroupResult:
    """Set the group of the index's sessions whose metadata matches; rewrite ``index_file``."""
    add_group_wrapper(index_file, opts.as_dict())
    return AddGroupResult(index_file=Path(index_file))


def get_best_model(model_dir: str, cp_path: str, output_file: str | None,
                   opts: BestModelOptions) -> BestModelResult:
    """Pick the model in ``model_dir`` closest to the changepoints in ``cp_path``; plot the comparison."""
    best_model_info, figure = get_best_fit_model_wrapper(
        model_dir, cp_path, output_file, opts.plot_all, opts.ext, opts.fps, opts.objective
    )
    return BestModelResult(best_model_info=best_model_info, figure=figure,
                           figure_files=_figures(output_file) if output_file is not None else [])


def make_df(model_fit: str, index_file: str, output_file: str | Path) -> MakeDfResult:
    """Write the syllable and scalar dataframe, and its statistics, as ``.csv`` or ``.parquet``."""
    output_file = Path(output_file)
    extension = output_file.suffix[1:]
    if extension not in ["csv", "parquet"]:
        raise ValueError(f"Extension {extension} is not supported. Please use .csv or .parquet.")
    make_df_wrapper(model_fit, index_file, output_file, extension)
    return MakeDfResult(dataframe_file=output_file,
                        stats_file=output_file.with_name(f"{output_file.stem}_stats.{extension}"))


def make_crowd_movies(index_file: str, model_path: str, output_dir: str, opts: CrowdMovieOptions, *,
                      shim_flat_config: Mapping | None = None) -> CrowdMoviesResult:
    """Write a crowd movie for each syllable into ``output_dir``.

    ``shim_flat_config`` (R2, removed by PR 4.1) is a notebook front end's whole
    settings dict, passed under the options.
    """
    settings = {**(shim_flat_config or {}), **opts.as_dict()}
    return CrowdMoviesResult(movies=make_crowd_movies_wrapper(index_file, model_path, output_dir, settings))


def plot_scalar_summary(index_file: str, output_file: str, opts: ScalarSummaryOptions) -> FiguresResult:
    """Plot each group's scalar summary to ``<output_file>_summary``."""
    plot_scalar_summary_wrapper(index_file, output_file, colors=opts.colors)
    return FiguresResult(figure_files=_figures(output_file, "_summary"))


def plot_group_position_heatmaps(index_file: str, output_file: str, opts: HeatmapOptions) -> FiguresResult:
    """Plot each group's mean position heatmap to ``output_file``."""
    plot_mean_group_position_pdf_wrapper(index_file, output_file, normalize=opts.normalize)
    return FiguresResult(figure_files=_figures(output_file))


def plot_verbose_position_heatmaps(index_file: str, output_file: str, opts: HeatmapOptions) -> FiguresResult:
    """Plot each session's position heatmap to ``output_file``."""
    plot_verbose_pdfs_wrapper(index_file, output_file, normalize=opts.normalize)
    return FiguresResult(figure_files=_figures(output_file))


def plot_transition_graph(index_file: str, model_fit: str, output_file: str,
                          opts: TransitionGraphOptions) -> FiguresResult:
    """Plot each group's transition graph to ``output_file``."""
    plot_transition_graph_wrapper(index_file, model_fit, output_file, opts.as_dict())
    return FiguresResult(figure_files=_figures(output_file))


def plot_stats(index_file: str, model_fit: str, output_file: str, opts: StatsPlotOptions) -> FiguresResult:
    """Plot a syllable statistic for each group to ``output_file``."""
    plot_syllable_stat_wrapper(model_fit, index_file, output_file, **opts.as_dict())
    return FiguresResult(figure_files=_figures(output_file))

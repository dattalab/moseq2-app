"""The analysis pipeline steps, one function each.

The ``moseq2-viz`` commands (``cli.py``) and the notebook's model selection
(``gui.py``) resolve their settings into one options object and call these. A
step needs no ``config.yaml``: from a script, pass the sessions, the paths and
an options object, such as ``plot_stats(session_index(Project.at("my_project")),
"model.p", "plots/usage", StatsPlotOptions(stat="duration"))``.

``index`` is the project's sessions with their groups as an index in memory
(``moseq2_common.bookkeeping.session_index``); a path to a 1.5.0
``moseq2-index.yaml`` is read the same way. Each step hands its work function
(the ``*_wrapper`` in ``moseq2_viz.helpers.wrappers``) the options, as keywords
or as the dict the wrapper takes.

``add_group`` writes ``config.yaml``'s ``groups``, and ``get_best_model``
records the model it picks as the project's current model in its
``moseq2-project.yaml``.

Every other step writes the record of its run (``moseq2_common.records``) as
a ``.yaml`` beside its output (``sidecar_path``): ``<figure>.yaml`` for a figure
named without ``.png``/``.pdf``, ``moseq_df.yaml`` for ``moseq_df.csv``,
``crowd_movies.yaml`` for the crowd-movie folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moseq2_common.bookkeeping import relative_path, session_index, update
from moseq2_common.project import Project, read_groups, write_groups
from moseq2_common.records import start_record, write_record
from moseq2_viz.helpers.wrappers import (
    add_group_wrapper,
    get_best_fit_model_wrapper,
    models_in,
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
    #: The sessions whose group was set, by key, with the group.
    changed: dict


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


def _start(step, opts, index, *files) -> Any:
    """The record of ``step``: its inputs ``files``, the sessions ``index`` names and the project's
    settings, when ``index`` is a project's sessions."""
    sessions, project = [], None
    if isinstance(index, dict):
        entries = index.get("files", [])
        sessions = [entry["path"][0] for entry in (entries.values() if isinstance(entries, dict) else entries)]
        if index.get("bookkeeping_file"):
            project = Project.at(Path(index["bookkeeping_file"]).parent)
    elif isinstance(index, (str, Path)):
        files = (index, *files)
    settings = [project.config_file] if project is not None and project.config_file.is_file() else []
    return start_record(step, opts if opts is not None else {}, [*settings, *[f for f in files if f], *sessions],
                        project=project)


def _finish(record, output, files):
    """Write ``record``, finished with ``files``, beside ``output``."""
    write_record(output, record.finish({"files": [str(f) for f in files if Path(f).exists()]}))


def add_group(project: Project, opts: AddGroupOptions) -> AddGroupResult:
    """Set the group of the project's sessions whose metadata matches; write ``config.yaml``'s ``groups``."""
    changed = add_group_wrapper(session_index(project, exclude=False), opts.as_dict())
    write_groups(project, {**read_groups(project), **changed})
    return AddGroupResult(changed=changed)


def get_best_model(model_dir: str, cp_path: str, output_file: str | None,
                   opts: BestModelOptions, *, project: Project | None = None) -> BestModelResult:
    """Pick the model in ``model_dir`` closest to the changepoints in ``cp_path``; plot the comparison.

    In a project, every model compared is listed among the project's fitted models (a kappa scan's fits,
    run as jobs, record nothing themselves), and the model picked becomes the current model.
    """
    record = _start("get-best-model", opts, None, model_dir, cp_path)
    best_model_info, figure = get_best_fit_model_wrapper(
        model_dir, cp_path, output_file, opts.plot_all, opts.ext, opts.objective
    )
    if output_file is not None:
        record.note("best_model", best_model_info.get(f"best model - {opts.objective}"))
        _finish(record, output_file, _figures(output_file))
    best = best_model_info.get(f"best model - {opts.objective}")
    if project is not None and best:
        compared = [relative_path(project, model) for model in sorted(models_in(model_dir, opts.ext))]
        model = relative_path(project, best)

        def change(bookkeeping):
            fitted = bookkeeping.models.setdefault("fitted", [])
            for name in [*compared, model]:
                if name not in fitted:
                    fitted.append(name)
            bookkeeping.models["current"] = model

        update(project, change)
    return BestModelResult(best_model_info=best_model_info, figure=figure,
                           figure_files=_figures(output_file) if output_file is not None else [])


def make_df(model_fit: str, index: dict | str, output_file: str | Path) -> MakeDfResult:
    """Write the syllable and scalar dataframe, and its statistics, as ``.csv`` or ``.parquet``."""
    output_file = Path(output_file)
    extension = output_file.suffix[1:]
    if extension not in ["csv", "parquet"]:
        raise ValueError(f"Extension {extension} is not supported. Please use .csv or .parquet.")
    record = _start("make-df", None, index, model_fit)
    make_df_wrapper(model_fit, index, output_file, extension)
    result = MakeDfResult(dataframe_file=output_file,
                          stats_file=output_file.with_name(f"{output_file.stem}_stats.{extension}"))
    _finish(record, output_file, [result.dataframe_file, result.stats_file])
    return result


def make_crowd_movies(index: dict | str, model_path: str, output_dir: str, opts: CrowdMovieOptions) -> CrowdMoviesResult:
    """Write a crowd movie for each syllable into ``output_dir``."""
    record = _start("make-crowd-movies", opts, index, model_path)
    result = CrowdMoviesResult(movies=make_crowd_movies_wrapper(index, model_path, output_dir, opts.as_dict()))
    _finish(record, output_dir, [movie for movies in result.movies.values() for movie in movies]
            if isinstance(result.movies, dict) else [])
    return result


def plot_scalar_summary(index: dict | str, output_file: str, opts: ScalarSummaryOptions) -> FiguresResult:
    """Plot each group's scalar summary to ``<output_file>_summary``."""
    record = _start("plot-scalar-summary", opts, index)
    plot_scalar_summary_wrapper(index, output_file, colors=opts.colors)
    _finish(record, f"{output_file}_summary", _figures(output_file, "_summary"))
    return FiguresResult(figure_files=_figures(output_file, "_summary"))


def plot_group_position_heatmaps(index: dict | str, output_file: str, opts: HeatmapOptions) -> FiguresResult:
    """Plot each group's mean position heatmap to ``output_file``."""
    record = _start("plot-group-position-heatmaps", opts, index)
    plot_mean_group_position_pdf_wrapper(index, output_file, normalize=opts.normalize)
    _finish(record, output_file, _figures(output_file))
    return FiguresResult(figure_files=_figures(output_file))


def plot_verbose_position_heatmaps(index: dict | str, output_file: str, opts: HeatmapOptions) -> FiguresResult:
    """Plot each session's position heatmap to ``output_file``."""
    record = _start("plot-verbose-position-heatmaps", opts, index)
    plot_verbose_pdfs_wrapper(index, output_file, normalize=opts.normalize)
    _finish(record, output_file, _figures(output_file))
    return FiguresResult(figure_files=_figures(output_file))


def plot_transition_graph(index: dict | str, model_fit: str, output_file: str,
                          opts: TransitionGraphOptions) -> FiguresResult:
    """Plot each group's transition graph to ``output_file``."""
    record = _start("plot-transition-graph", opts, index, model_fit)
    plot_transition_graph_wrapper(index, model_fit, output_file, opts.as_dict())
    _finish(record, output_file, _figures(output_file))
    return FiguresResult(figure_files=_figures(output_file))


def plot_stats(index: dict | str, model_fit: str, output_file: str, opts: StatsPlotOptions) -> FiguresResult:
    """Plot a syllable statistic for each group to ``output_file``."""
    record = _start("plot-stats", opts, index, model_fit)
    plot_syllable_stat_wrapper(model_fit, index, output_file, **opts.as_dict())
    _finish(record, output_file, _figures(output_file))
    return FiguresResult(figure_files=_figures(output_file))

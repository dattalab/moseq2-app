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
``crowd_movies.yaml`` for the crowd-movie folder. Figures and dataframes are
written with their record under temporary names and renamed into place when
complete (``atomic_outputs``); a crowd-movie folder's record is removed before
its movies are rendered and written when they are all done.

Before any work, a step stops on a model file, a changepoints file or a
session's extraction that does not exist, is empty or cannot be read
(``moseq2_common.records.check_inputs``).

Crowd movies are a cache: the syllable labeler reuses a folder of them only
while its record shows they were rendered from the same model (the run its
record names), with the same ``crowd_movies`` settings and from the same sessions
(``crowd_movies_out_of_date``), and otherwise renders them again and says why
(``rerender_message``). The record is read to check, never to render with.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moseq2_common.bookkeeping import SESSION_ADVICE, relative_path, session_index, update
from moseq2_common.project import Project, read_groups, write_groups
from moseq2_common.records import (atomic_outputs, check_inputs, read_record, record_text, sidecar_path,
                                   start_record, unchanged, write_record)
from moseq2_common.yaml import read_yaml
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


def _sessions(index) -> tuple[list[str], Project | None]:
    """The extraction files of the sessions ``index`` names, and the project it was read from."""
    sessions, project = [], None
    if isinstance(index, dict):
        entries = index.get("files", [])
        sessions = [entry["path"][0] for entry in (entries.values() if isinstance(entries, dict) else entries)]
        if index.get("bookkeeping_file"):
            project = Project.at(Path(index["bookkeeping_file"]).parent)
    return sessions, project


def _start(step, opts, index, *files) -> Any:
    """The record of ``step``: its inputs ``files``, the sessions ``index`` names and the project's
    settings, when ``index`` is a project's sessions."""
    sessions, project = _sessions(index)
    if isinstance(index, (str, Path)):
        files = (index, *files)
    settings = [project.config_file] if project is not None and project.config_file.is_file() else []
    return start_record(step, opts if opts is not None else {}, [*settings, *[f for f in files if f], *sessions],
                        project=project)


def _finish(record, output, files):
    """Write ``record``, finished with ``files``, beside ``output``."""
    write_record(output, record.finish({"files": [str(f) for f in files if Path(f).exists()]}))


def _figures_written(record, output_file, write, suffix=""):
    """Run ``write(stem)``, which writes the figure ``<stem><suffix>.png`` and ``.pdf``, under a temporary stem;
    write ``record`` beside them and rename all three to ``output_file``'s name when complete
    (``atomic_outputs``)."""
    final = Path(f"{output_file}{suffix}")
    with atomic_outputs(output_file, (f"{suffix}.png", f"{suffix}.pdf", f"{suffix}.yaml")) as temporary:
        write(str(temporary))
        partial = Path(f"{temporary}{suffix}")
        written = [final.with_name(f"{final.name}{ext}") for ext in (".png", ".pdf")
                   if partial.with_name(f"{partial.name}{ext}").exists()]
        write_record(partial, record.finish({"files": [str(f) for f in written]}))


#: What to do about each kind of file an analysis step reads, when it is unusable.
MODEL_ADVICE = ("Fit the model again (moseq2-model learn-model, or the notebook's modeling cell), or give the "
                "path of a model file that exists.")
CHANGEPOINTS_ADVICE = ("Compute the changepoints again (moseq2-pca compute-changepoints, or the notebook's "
                       "changepoints cell), or give the changepoints file it wrote.")


def _check(step, index, *files):
    """Stop ``step`` before its work when one of ``files`` (each with what to do about it) or a session's
    extraction ``index`` names does not exist, is empty or cannot be read."""
    check_inputs(step, {**dict(files), **{h5: SESSION_ADVICE for h5 in _sessions(index)[0]}})


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
    _check("get-best-model", None, (model_dir, MODEL_ADVICE), (cp_path, CHANGEPOINTS_ADVICE))
    record = _start("get-best-model", opts, None, model_dir, cp_path)
    found = {}

    def compare(stem):
        found["info"], found["figure"] = get_best_fit_model_wrapper(
            model_dir, cp_path, stem, opts.plot_all, opts.ext, opts.objective
        )
        if stem is not None:
            record.note("best_model", found["info"].get(f"best model - {opts.objective}"))

    if output_file is None:
        compare(None)
    else:
        _figures_written(record, output_file, compare)
    best_model_info, figure = found["info"], found["figure"]
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
    _check("make-df", index, (model_fit, MODEL_ADVICE))
    record = _start("make-df", None, index, model_fit)
    result = MakeDfResult(dataframe_file=output_file,
                          stats_file=output_file.with_name(f"{output_file.stem}_stats.{extension}"))
    with atomic_outputs(output_file.with_suffix(""), (f".{extension}", f"_stats.{extension}", ".yaml")) as temporary:
        partial = Path(f"{temporary}.{extension}")
        make_df_wrapper(model_fit, index, partial, extension)
        write_record(partial, record.finish({"files": [str(result.dataframe_file), str(result.stats_file)]}))
    return result


def make_crowd_movies(index: dict | str, model_path: str, output_dir: str, opts: CrowdMovieOptions) -> CrowdMoviesResult:
    """Write a crowd movie for each syllable into ``output_dir``."""
    _check("make-crowd-movies", index, (model_path, MODEL_ADVICE))
    record = _start("make-crowd-movies", opts, index, model_path)
    # movies rendered partway never carry a record, so the labeler renders them again
    sidecar_path(output_dir).unlink(missing_ok=True)
    result = CrowdMoviesResult(movies=make_crowd_movies_wrapper(index, model_path, output_dir, opts.as_dict()))
    _finish(record, output_dir, [movie for movies in result.movies.values() for movie in movies]
            if isinstance(result.movies, dict) else [])
    return result


#: Crowd-movie settings that say how the movies are rendered or which syllables get one, not what a movie
#: shows; the labeler checks for a movie of each syllable it offers on its own.
_NOT_COMPARED = ("processes", "progress_bar", "max_syllable", "specific_syllable")


def _as_recorded(values: dict) -> dict:
    """``values`` as a record holds them once read back (tuples as lists)."""
    return read_yaml(io.StringIO(record_text(values))) or {}


def crowd_movies_out_of_date(output_dir: str, model_path: str, index: dict, opts: CrowdMovieOptions) -> list[str]:
    """Why the crowd movies in ``output_dir`` cannot be shown for ``model_path``, the sessions of ``index`` and
    ``opts``; an empty list when they can.

    The folder's record (``crowd_movies.yaml`` beside it) names what the movies were rendered from: the model
    with the run that fit it, the settings and the sessions. Files are matched by their path in the project and
    the model by its run (``unchanged``), never by file time, so a project copied elsewhere keeps its movies.
    """
    try:
        record = read_record(output_dir)
    except FileNotFoundError:
        return ["there is no record of what they were rendered from"]
    sessions, project = _sessions(index)

    def where(path):
        if project is None:
            return os.path.abspath(path)
        return Path(os.path.relpath(os.path.abspath(path), Path(project.folder).resolve())).as_posix()

    inputs = {entry.get("relative_path") or entry.get("path"): entry for entry in record.get("inputs") or []
              if isinstance(entry, dict)}
    reasons = []
    model = inputs.get(where(model_path))
    if model is None or not unchanged(model, model_path):
        reasons.append("the model changed since they were made")
    recorded, current = record.get("options") or {}, _as_recorded(opts.as_dict())
    changed = sorted(name for name in set(recorded) | set(current)
                     if name not in _NOT_COMPARED and recorded.get(name) != current.get(name))
    if changed:
        reasons.append(f"the crowd_movies settings changed: {', '.join(changed)}")
    config = [where(project.config_file)] if project is not None else []
    rendered_from = set(inputs) - {where(model_path), *config}
    if rendered_from != {where(session) for session in sessions}:
        reasons.append("the sessions changed")
    return reasons


def rerender_message(output_dir: str, reasons: list[str]) -> str:
    """What the labeler prints before rendering the crowd movies in ``output_dir`` again, for ``reasons``."""
    return f"Rendering the crowd movies in {str(output_dir).rstrip(os.sep + '/')} again: {'; '.join(reasons)}."


def plot_scalar_summary(index: dict | str, output_file: str, opts: ScalarSummaryOptions) -> FiguresResult:
    """Plot each group's scalar summary to ``<output_file>_summary``."""
    _check("plot-scalar-summary", index)
    record = _start("plot-scalar-summary", opts, index)
    _figures_written(record, output_file,
                     lambda stem: plot_scalar_summary_wrapper(index, stem, colors=opts.colors), "_summary")
    return FiguresResult(figure_files=_figures(output_file, "_summary"))


def plot_group_position_heatmaps(index: dict | str, output_file: str, opts: HeatmapOptions) -> FiguresResult:
    """Plot each group's mean position heatmap to ``output_file``."""
    _check("plot-group-position-heatmaps", index)
    record = _start("plot-group-position-heatmaps", opts, index)
    _figures_written(record, output_file,
                     lambda stem: plot_mean_group_position_pdf_wrapper(index, stem, normalize=opts.normalize))
    return FiguresResult(figure_files=_figures(output_file))


def plot_verbose_position_heatmaps(index: dict | str, output_file: str, opts: HeatmapOptions) -> FiguresResult:
    """Plot each session's position heatmap to ``output_file``."""
    _check("plot-verbose-position-heatmaps", index)
    record = _start("plot-verbose-position-heatmaps", opts, index)
    _figures_written(record, output_file,
                     lambda stem: plot_verbose_pdfs_wrapper(index, stem, normalize=opts.normalize))
    return FiguresResult(figure_files=_figures(output_file))


def plot_transition_graph(index: dict | str, model_fit: str, output_file: str,
                          opts: TransitionGraphOptions) -> FiguresResult:
    """Plot each group's transition graph to ``output_file``."""
    _check("plot-transition-graph", None, (model_fit, MODEL_ADVICE))  # it reads the model's labels only
    record = _start("plot-transition-graph", opts, index, model_fit)
    _figures_written(record, output_file,
                     lambda stem: plot_transition_graph_wrapper(index, model_fit, stem, opts.as_dict()))
    return FiguresResult(figure_files=_figures(output_file))


def plot_stats(index: dict | str, model_fit: str, output_file: str, opts: StatsPlotOptions) -> FiguresResult:
    """Plot a syllable statistic for each group to ``output_file``."""
    _check("plot-stats", index, (model_fit, MODEL_ADVICE))
    record = _start("plot-stats", opts, index, model_fit)
    _figures_written(record, output_file,
                     lambda stem: plot_syllable_stat_wrapper(model_fit, index, stem, **opts.as_dict()))
    return FiguresResult(figure_files=_figures(output_file))

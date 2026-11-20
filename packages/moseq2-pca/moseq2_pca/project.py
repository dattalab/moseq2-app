"""The PCA steps in a project: the project layer.

Each step function here (``train_pca``, ``apply_pca``, ``compute_changepoints``)
is one step run in a project. It takes the project
(``moseq2_common.project.Project``) first, then what a person can type for the
step (the folder of extractions to read instead of the project's sessions, the
output folder and name, the components and scores files) and the step's
resolved options object, and returns the step's result. It resolves what the
project decides for the step:

- the sessions it reads: those the project's ``moseq2-project.yaml`` lists
  (``moseq2_common.bookkeeping.session_index``), the same list the model and
  analysis steps use, or the extractions under a typed folder, matched to
  session keys (``moseq2_common.bookkeeping.extraction_keys``); either way less
  the sessions ``config.yaml``'s ``exclude`` lists, or, when the step is given
  ``exclude``, the ones it names (``[]`` for none) for that run, and in the order
  they were recorded (``moseq2_common.bookkeeping.recording_order``);
- the components and scores it reads: those ``moseq2-project.yaml`` names
  (``recorded_paths``), unless a file is given;
- that the components were trained with the settings ``config.yaml``'s ``pca``
  section sets (``check_trained_settings``).

It starts the run's record with the project (``start_record``), so the record
names the project folder, lists ``config.yaml`` and describes each input relative
to the project folder, and notes the sessions used and left out. It calls the
file layer (``moseq2_pca.steps``) with the sessions as (session key, extraction)
pairs and that record, and records the file the step wrote in
``moseq2-project.yaml`` as the project's components, scores or changepoints, where
the next step finds it. No step writes ``config.yaml``.

When ``train_pca`` or ``apply_pca`` would write over an output that already
exists and its ``overwrite_pca_*`` option is off, the question is the first
thing decided: before the sessions are listed, any line is printed or any input
is checked, the step raises ``moseq2_common.project.OverwriteNeeded``. The caller
asks and calls again with the answer as ``replace_existing``, which leaves the
options, and so the record, as they were resolved.

``folder_sessions`` with no project serves the command line outside any
project: it names each extraction under the folder by the project that owns it,
or by its path when none does, and leaves out the ones a typed ``exclude`` names.

The file layer is reached through its module (``steps.train_pca``), never by a
name imported from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

from moseq2_common import records
from moseq2_common.bookkeeping import (BookkeepingError, check_given_exclude, exclude_line, extraction_keys, load,
                                       relative_path, resolve_path, session_index, sessions_folder, update)
from moseq2_common.options import read_config
from moseq2_common.project import Project, read_exclude, settings_files
from moseq2_common.records import InputError, Record
from moseq2_pca import steps
from moseq2_pca.helpers.data import TRAINED_SETTINGS, TrainedSettingsError, read_trained_settings
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions
from moseq2_pca.util import find_extractions

__all__ = ["PCASessions", "apply_pca", "check_trained_settings", "compute_changepoints", "folder_sessions",
           "project_sessions", "recorded_paths", "start_record", "train_pca", "trained_settings_heads_up"]


@dataclass(frozen=True)
class PCASessions:
    #: (session key, extraction ``.h5``) of each session read, in the order to read them (recording order).
    pairs: list[tuple[str, str]]
    #: The session keys ``exclude`` left out.
    left_out: list[str]


def project_sessions(project: Project, exclude: list[str] | None = None) -> PCASessions:
    """The sessions ``project``'s bookkeeping file lists, less those ``exclude`` names (None: ``config.yaml``'s
    ``exclude``)."""
    sessions_folder(project)  # stops when the project lists no sessions yet
    index = session_index(project, exclude=exclude)
    files = index["files"]
    if not files:
        if exclude is not None:
            raise BookkeepingError(
                f"Every session in {project.bookkeeping_file} is in the list to leave out given for this run, so "
                "there is nothing for PCA to read. Name fewer sessions to leave out."
            )
        raise BookkeepingError(
            f"Every session in {project.bookkeeping_file} is listed under exclude in {project.config_file}, so "
            f"there is nothing for PCA to read. Remove at least one session from exclude in {project.config_file}."
        )
    return PCASessions([(f["key"], f["path"][0]) for f in files], list(index.get("excluded") or {}))


def folder_sessions(step: str, input_dir: str, project: Project | None,
                    exclude: list[str] | None = None) -> PCASessions:
    """The extractions under ``input_dir``, less those whose session ``exclude`` names (None: the running
    ``project``'s ``config.yaml``'s ``exclude``, none outside a project).

    Each is named by its session in ``project``, else in the project that holds it, else by its path relative to
    ``input_dir`` (``extraction_keys``). Stops ``step`` when the folder does not exist, when it holds no
    extraction, and when ``exclude`` leaves none.
    """
    if not Path(input_dir).is_dir():
        raise InputError(
            f"{step} needs the folder of extractions {abspath(input_dir)}, which does not exist. Check the path "
            "given with --input-dir, or run it from the project folder (or name the project with --config-file "
            "/path/to/project/config.yaml) to read the sessions its moseq2-project.yaml lists."
        )
    h5s, dicts, _ = find_extractions(input_dir)
    keys = extraction_keys(h5s, [info["uuid"] for info in dicts], project, input_dir)
    if exclude is None:
        listed = read_exclude(project) if project is not None else []
    else:
        listed = list(dict.fromkeys(str(key) for key in exclude))
        check_given_exclude(listed, keys, f"the extractions in {abspath(input_dir)}")
    left_out = [key for key in keys if key in listed]
    line = exclude_line(left_out, given=exclude is not None, in_project=project is not None)
    if line:
        print(line)
    if not left_out and exclude is None:
        if not h5s:
            raise InputError(
                f"{step} found no extractions in {abspath(input_dir)}: it looks for the .h5 files extraction "
                "writes (results.h5, holding frames and a session uuid) in that folder and every folder below it. "
                "Run it from the project folder, or name the project with --config-file "
                "/path/to/project/config.yaml, to read the sessions its moseq2-project.yaml lists; or pass the "
                "folder holding the extractions with --input-dir."
            )
        return PCASessions(list(zip(keys, h5s)), [])
    kept = [(key, h5) for h5, key in zip(h5s, keys) if key not in left_out]
    if not kept:
        raise InputError(
            f"Every extraction in {abspath(input_dir)} is left out ({', '.join(left_out)}), so there is nothing for "
            "PCA to read. Name fewer sessions to leave out, or pass another folder with --input-dir."
        )
    return PCASessions(kept, left_out)


def start_record(step: str, opts, sessions: PCASessions, *files: str | None, project: Project | None) -> Record:
    """The record of ``step``: its inputs ``project``'s ``config.yaml`` (none outside a project), the ``files``
    that are not None, then the extractions of ``sessions``, noting the sessions used and those left out."""
    settings = settings_files(project, session_config=False) if project is not None else []
    record = records.start_record(
        step, opts, [*settings, *[f for f in files if f], *steps.extraction_inputs(sessions.pairs)], project=project)
    record.note("sessions", [key for key, _ in sessions.pairs])
    record.note("exclude", sessions.left_out)
    return record


def recorded_paths(project: Project | None) -> dict[str, str]:
    """The components, scores and changepoints files the project's bookkeeping file names (``pca_file``,
    ``pca_file_scores``, ``changepoint_file``); empty outside a project."""
    if project is None:
        return {}
    stored = load(project).pca
    names = {'components': 'pca_file', 'scores': 'pca_file_scores', 'changepoints': 'changepoint_file'}
    return {names[output]: str(resolve_path(project, path)) for output, path in stored.items() if output in names}


def _shown(value):
    """A setting's value as config.yaml writes it."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shown(v) for v in value) + "]"
    return str(value)


def _same(setting, trained, configured):
    """Whether two values of ``setting`` are the same once read as ``train-pca`` reads them."""
    def plain(value):
        value = getattr(TrainPCAOptions.resolve(config={setting: value}), setting)
        return list(value) if isinstance(value, tuple) else value
    return plain(trained) == plain(configured)


def check_trained_settings(components_file: str, project: Project) -> None:
    """Stop when ``project``'s ``config.yaml`` sets a setting the components were trained with to another value.

    Each of ``TRAINED_SETTINGS`` that the ``pca`` section sets is compared with the value the components in
    ``components_file`` carry. Nothing is checked when either file does not exist yet (the step itself then
    says which is missing).
    """
    if not Path(components_file).is_file() or not Path(project.config_file).is_file():
        return
    section = read_config(project.config_file).get("pca") or {}
    configured = {key: section[key] for key in TRAINED_SETTINGS if section.get(key) is not None}
    if not configured:
        return
    trained = read_trained_settings(components_file)
    differ = [
        f"The PCA components in {components_file} were trained with {key} {_shown(trained[key])}, but the pca "
        f"section of {project.config_file} sets {_shown(value)}. Scores and changepoints must use the trained "
        f"value: set {key} back to {_shown(trained[key])} in {project.config_file}, or train PCA again with these "
        "settings."
        for key, value in configured.items() if not _same(key, trained[key], value)
    ]
    if differ:
        raise TrainedSettingsError("\n".join(differ))


def trained_settings_heads_up(typed: dict, configured: dict, config_file) -> list[str]:
    """One line for each of ``TRAINED_SETTINGS`` typed for ``train-pca`` (``typed``) at another value than
    ``config_file``'s ``pca`` section (``configured``) sets, saying that ``apply-pca`` and ``compute-changepoints``
    will stop (``check_trained_settings``) until the two agree."""
    return [
        f"train-pca trains with {key} {_shown(typed[key])}, typed on the command line, but the pca section of "
        f"{config_file} sets {_shown(configured[key])}. apply-pca and compute-changepoints will stop until the two "
        f"agree: set {key} to {_shown(typed[key])} in {config_file}, or train again without typing it."
        for key in TRAINED_SETTINGS
        if key in typed and configured.get(key) is not None and not _same(key, typed[key], configured[key])
    ]


def _sessions(step, project, input_dir, exclude):
    """What ``step`` reads in ``project``: the extractions under ``input_dir`` when one is given, else the
    project's sessions."""
    if input_dir is not None:
        return folder_sessions(step, input_dir, project, exclude)
    return project_sessions(project, exclude)


def _record(project, output, written):
    """Record ``written`` as the project's PCA ``output`` (``components``, ``scores``, ``changepoints``)."""
    if written is None:
        return

    def change(bookkeeping):
        bookkeeping.pca[output] = relative_path(project, written)

    update(project, change)


def train_pca(project: Project, input_dir: str | None, output_dir: str, output_file: str, opts: TrainPCAOptions, *,
              exclude: list[str] | None = None, replace_existing: bool | None = None) -> steps.TrainPCAResult:
    """Train PCA in ``project`` on the extractions under ``input_dir``, else on the project's sessions, less the
    sessions ``exclude`` names (None: ``config.yaml``'s ``exclude``); write ``<output_dir>/<output_file>.h5`` and
    record it as the project's components.

    When that file exists and ``opts.overwrite_pca_train`` is off, ``replace_existing`` decides (None: raise
    ``OverwriteNeeded`` before anything else).
    """
    if steps.keep_existing(steps.output_path(output_dir, output_file), opts.overwrite_pca_train, replace_existing,
                           step="train_pca", option="overwrite_pca_train"):
        return steps.TrainPCAResult(components_file=None)
    sessions = _sessions("train-pca", project, input_dir, exclude)
    steps.check_extractions("train-pca", sessions.pairs)
    record = start_record("train-pca", opts, sessions, project=project)
    result = steps.train_pca(sessions.pairs, output_dir, output_file, opts, record=record, replace_existing=True)
    _record(project, "components", result.components_file)
    return result


def apply_pca(project: Project, input_dir: str | None, output_dir: str, output_file: str, opts: ApplyPCAOptions, *,
              components_file: str | None = None, exclude: list[str] | None = None,
              replace_existing: bool | None = None) -> steps.ApplyPCAResult:
    """Compute the PC scores in ``project`` of the extractions under ``input_dir``, else of the project's
    sessions, less the sessions ``exclude`` names (None: ``config.yaml``'s ``exclude``), with the components in
    ``components_file`` (None: those the project recorded, else ``<output_dir>/pca.h5``); write
    ``<output_dir>/<output_file>.h5`` and record it as the project's scores.

    Stops when ``config.yaml``'s ``pca`` section sets a setting the components were trained with to another
    value. When the output exists and ``opts.overwrite_pca_apply`` is off, ``replace_existing`` decides (None:
    raise ``OverwriteNeeded`` before anything else).
    """
    if steps.keep_existing(steps.output_path(output_dir, output_file), opts.overwrite_pca_apply, replace_existing,
                           step="apply_pca", option="overwrite_pca_apply"):
        return steps.ApplyPCAResult(scores_file=None)
    components_file = components_file or recorded_paths(project).get('pca_file')
    sessions = _sessions("apply-pca", project, input_dir, exclude)
    components = components_file or join(abspath(output_dir), 'pca.h5')
    steps.check_extractions("apply-pca", sessions.pairs, (components, steps.COMPONENTS_ADVICE))
    check_trained_settings(components, project)
    record = start_record("apply-pca", opts, sessions, components_file, project=project)
    result = steps.apply_pca(sessions.pairs, output_dir, output_file, opts, components_file=components_file,
                             components_advice=steps.COMPONENTS_ADVICE, record=record, replace_existing=True)
    _record(project, "scores", result.scores_file)
    return result


def compute_changepoints(project: Project, input_dir: str | None, output_dir: str, output_file: str,
                         opts: ChangepointsOptions, *, components_file: str | None = None,
                         scores_file: str | None = None,
                         exclude: list[str] | None = None) -> steps.ChangepointsResult:
    """Compute the model-free changepoints in ``project`` of the extractions under ``input_dir``, else of the
    project's sessions, less the sessions ``exclude`` names (None: ``config.yaml``'s ``exclude``), with the
    components in ``components_file`` and the scores in ``scores_file`` (None: those the project recorded); write
    ``<output_dir>/<output_file>.h5`` and record it as the project's changepoints.

    Stops when ``config.yaml``'s ``pca`` section sets a setting the components were trained with to another
    value.
    """
    recorded = recorded_paths(project)
    components_file = components_file or recorded.get('pca_file')
    scores_file = scores_file or recorded.get('pca_file_scores')
    sessions = _sessions("compute-changepoints", project, input_dir, exclude)
    components = components_file or join(abspath(output_dir), 'pca.h5')
    steps.check_extractions("compute-changepoints", sessions.pairs, (components, steps.COMPONENTS_ADVICE))
    check_trained_settings(components, project)
    record = start_record("compute-changepoints", opts, sessions, components_file, scores_file, project=project)
    result = steps.compute_changepoints(sessions.pairs, output_dir, output_file, opts,
                                        components_file=components_file, scores_file=scores_file,
                                        components_advice=steps.COMPONENTS_ADVICE, record=record)
    _record(project, "changepoints", result.changepoints_file)
    return result

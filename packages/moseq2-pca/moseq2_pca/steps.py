"""The PCA pipeline steps, one function each.

``train-pca``, ``apply-pca`` and ``compute-changepoints`` on the command line
(``cli.py``) and the notebook's PCA cells (``gui.py``) resolve their settings
into one options object and call these. A step needs no ``config.yaml``: from a
script, pass the paths and an options object, such as
``train_pca("aggregate_results", "_pca", "pca", TrainPCAOptions(nworkers=4))``.

The sessions: inside a project (``project``), a step reads the sessions the
project's bookkeeping file lists (``moseq2_common.bookkeeping.session_index``),
the same list the model and analysis steps use, leaving out the ones
``config.yaml``'s ``exclude`` lists, and ``input_dir`` is not read. Outside a
project, it reads every extraction it finds under ``input_dir``. Either way the
sessions are read in the order they were recorded
(``moseq2_common.bookkeeping.recording_order``); the changepoints depend on
that order, since they draw one random projection per session in turn.

Each step hands its work function (the ``*_wrapper`` in
``moseq2_pca.helpers.wrappers``) the options as a dict, with the files it reads:
``components_file`` (the trained components; ``<output_dir>/pca.h5`` when None)
and, for changepoints, ``scores_file`` (the PC scores, read for missing-data
PCA).

Inside a project (``project``), each step records the file it wrote in the
project's bookkeeping file (``moseq2_common.bookkeeping``), where the next step
finds it: the components, the scores, the changepoints. No step writes
``config.yaml``.

Each ``.h5`` a step writes holds the record of the run (``moseq2_common.records``)
as its ``/provenance`` attribute: the options, every extraction and file read, and
the arrays written, with the figures written beside it.

The components carry the settings they were trained with (``TRAINED_SETTINGS``:
the filters, the height range, FFT and missing-data PCA), as their ``.h5``'s
``/settings`` attribute. ``apply_pca`` and ``compute_changepoints`` prepare
each session's frames with those values, and, inside a project, stop before any
work when ``config.yaml``'s ``pca`` section sets one of them to another value
(``check_trained_settings``).
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

import h5py

from moseq2_common.bookkeeping import BookkeepingError, relative_path, session_index, sessions_folder, update
from moseq2_common.options import read_config
from moseq2_common.project import Project
from moseq2_common.records import start_record, write_h5_record
from moseq2_pca.helpers.data import TRAINED_SETTINGS, TrainedSettingsError, read_trained_settings
from moseq2_pca.helpers.wrappers import apply_pca_wrapper, compute_changepoints_wrapper, train_pca_wrapper
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions
from moseq2_pca.util import find_extractions


@dataclass(frozen=True)
class TrainPCAResult:
    #: The components file; None when training wrote none (an existing file was kept, or training stopped).
    components_file: Path | None


@dataclass(frozen=True)
class ApplyPCAResult:
    #: The scores file; None when an existing one was kept (overwrite declined).
    scores_file: Path | None


@dataclass(frozen=True)
class ChangepointsResult:
    changepoints_file: Path


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


def check_trained_settings(components_file, project: Project | None) -> None:
    """Stop when ``project``'s ``config.yaml`` sets a setting the components were trained with to another value.

    Each of ``TRAINED_SETTINGS`` that the ``pca`` section sets is compared with the value the components in
    ``components_file`` carry. Nothing is checked outside a project, or when either file does not exist yet
    (the step itself then says which is missing).
    """
    if project is None or not Path(components_file).is_file() or not Path(project.config_file).is_file():
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


def _components(components_file, output_dir):
    """The components a step reads: ``components_file``, else ``<output_dir>/pca.h5``."""
    return components_file or join(abspath(output_dir), 'pca.h5')


def _written(output_dir, output_file):
    """The file a wrapper writes for ``output_dir`` and ``output_file``."""
    return Path(join(abspath(output_dir), f'{output_file}.h5'))


def _sessions(input_dir, project):
    """What a step reads: the project's sessions' ``.h5`` files, or ``input_dir`` outside a project."""
    if project is None:
        return input_dir
    sessions_folder(project)  # stops when the project lists no sessions yet
    files = session_index(project)["files"]
    if not files:
        raise BookkeepingError(
            f"Every session in {project.bookkeeping_file} is listed under exclude in {project.config_file}, so "
            f"there is nothing for PCA to read. Remove at least one session from exclude in {project.config_file}."
        )
    return [f["path"][0] for f in files]


def _start(step, opts, sessions, project, *files):
    """The record of ``step``, its inputs the extractions of ``sessions`` and ``files``."""
    extractions = [{"path": h5, "uuid": info["uuid"]} for h5, info in zip(*find_extractions(sessions)[:2])]
    settings = [path for path in (project.config_file,) if path.is_file()] if project is not None else []
    return start_record(step, opts, [*settings, *[f for f in files if f], *extractions], project=project)


def _write_record(record, written, figures=()):
    """Finish ``record`` with the arrays in ``written`` and the ``figures`` beside it, and store it there."""
    if written is None or not Path(written).is_file():
        return
    with h5py.File(written, "a") as f:
        arrays = {}
        f.visititems(lambda name, node: arrays.setdefault(name, node)
                     if isinstance(node, h5py.Dataset) and not name.startswith("metadata") else None)
        files = [str(written), *[str(p) for p in figures if Path(p).is_file()]]
        write_h5_record(f, record.finish({**arrays, "files": files}))


def _record(project, output, written):
    """Record ``written`` as the project's PCA ``output`` (``components``, ``scores``, ``changepoints``)."""
    if project is None or written is None:
        return

    def change(bookkeeping):
        bookkeeping.pca[output] = relative_path(project, written)

    update(project, change)


def train_pca(input_dir: str | None, output_dir: str, output_file: str, opts: TrainPCAOptions, *,
              project: Project | None = None) -> TrainPCAResult:
    """Train PCA on the sessions (the project's, else the extractions under ``input_dir``); write
    ``<output_dir>/<output_file>.h5``."""
    sessions = _sessions(input_dir, project)
    record = _start("train-pca", opts, sessions, project)
    config_data = train_pca_wrapper(sessions, opts.as_dict(), output_dir, output_file)
    pca_file = config_data.get('pca_file')
    result = TrainPCAResult(components_file=Path(pca_file) if pca_file else None)
    stem = str(_written(output_dir, output_file).with_suffix(""))
    _write_record(record, result.components_file,
                  [f"{stem}{name}" for name in ("_components.png", "_components.pdf", "_scree.png", "_scree.pdf")])
    _record(project, "components", result.components_file)
    return result


def apply_pca(input_dir: str | None, output_dir: str, output_file: str, opts: ApplyPCAOptions, *,
              components_file: str | None = None, project: Project | None = None) -> ApplyPCAResult:
    """Compute each session's PC scores with the components in ``components_file``, preparing the frames
    with the settings the components were trained with; write ``<output_dir>/<output_file>.h5``."""
    check_trained_settings(_components(components_file, output_dir), project)
    sessions = _sessions(input_dir, project)
    record = _start("apply-pca", opts, sessions, project, components_file)
    _, success = apply_pca_wrapper(sessions, {**opts.as_dict(), 'pca_file': components_file},
                                   output_dir, output_file)
    result = ApplyPCAResult(scores_file=_written(output_dir, output_file) if success else None)
    _write_record(record, result.scores_file)
    _record(project, "scores", result.scores_file)
    return result


def compute_changepoints(input_dir: str | None, output_dir: str, output_file: str, opts: ChangepointsOptions, *,
                         components_file: str | None = None,
                         scores_file: str | None = None, project: Project | None = None) -> ChangepointsResult:
    """Compute the model-free changepoints, with missing-data PCA when the components were trained with it;
    write ``<output_dir>/<output_file>.h5``."""
    check_trained_settings(_components(components_file, output_dir), project)
    sessions = _sessions(input_dir, project)
    record = _start("compute-changepoints", opts, sessions, project, components_file, scores_file)
    compute_changepoints_wrapper(
        sessions, {**opts.as_dict(), 'pca_file': components_file, 'pca_file_scores': scores_file},
        output_dir, output_file)
    result = ChangepointsResult(changepoints_file=_written(output_dir, output_file))
    stem = str(result.changepoints_file.with_suffix(""))
    _write_record(record, result.changepoints_file, [f"{stem}_dist.png", f"{stem}_dist.pdf"])
    _record(project, "changepoints", result.changepoints_file)
    return result

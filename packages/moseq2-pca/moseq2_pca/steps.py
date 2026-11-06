"""The PCA pipeline steps, one function each.

``train-pca``, ``apply-pca`` and ``compute-changepoints`` on the command line
(``cli.py``) and the notebook's PCA cells (``gui.py``) resolve their settings
into one options object and call these. A step needs no ``config.yaml``: from a
script, pass the paths and an options object, such as
``train_pca("aggregate_results", "_pca", "pca", TrainPCAOptions(nworkers=4))``.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

import h5py

from moseq2_common.bookkeeping import relative_path, update
from moseq2_common.project import Project
from moseq2_common.records import start_record, write_h5_record
from moseq2_pca.helpers.wrappers import apply_pca_wrapper, compute_changepoints_wrapper, train_pca_wrapper
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions
from moseq2_pca.util import recursive_find_h5s


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


def _written(output_dir, output_file):
    """The file a wrapper writes for ``output_dir`` and ``output_file``."""
    return Path(join(abspath(output_dir), f'{output_file}.h5'))


def _start(step, opts, input_dir, project, *files):
    """The record of ``step``, its inputs every extraction under ``input_dir`` and ``files``."""
    extractions = [{"path": h5, "uuid": info["uuid"]} for h5, info in zip(*recursive_find_h5s(input_dir)[:2])]
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


def train_pca(input_dir: str, output_dir: str, output_file: str, opts: TrainPCAOptions, *,
              project: Project | None = None) -> TrainPCAResult:
    """Train PCA on the extractions under ``input_dir``; write ``<output_dir>/<output_file>.h5``."""
    record = _start("train-pca", opts, input_dir, project)
    config_data = train_pca_wrapper(input_dir, opts.as_dict(), output_dir, output_file)
    pca_file = config_data.get('pca_file')
    result = TrainPCAResult(components_file=Path(pca_file) if pca_file else None)
    stem = str(_written(output_dir, output_file).with_suffix(""))
    _write_record(record, result.components_file,
                  [f"{stem}{name}" for name in ("_components.png", "_components.pdf", "_scree.png", "_scree.pdf",
                                                ".yaml")])
    _record(project, "components", result.components_file)
    return result


def apply_pca(input_dir: str, output_dir: str, output_file: str, opts: ApplyPCAOptions, *,
              components_file: str | None = None, project: Project | None = None) -> ApplyPCAResult:
    """Compute each extraction's PC scores with the components in ``components_file``; write
    ``<output_dir>/<output_file>.h5``."""
    record = _start("apply-pca", opts, input_dir, project, components_file)
    _, success = apply_pca_wrapper(input_dir, {**opts.as_dict(), 'pca_file': components_file},
                                   output_dir, output_file)
    result = ApplyPCAResult(scores_file=_written(output_dir, output_file) if success else None)
    _write_record(record, result.scores_file)
    _record(project, "scores", result.scores_file)
    return result


def compute_changepoints(input_dir: str, output_dir: str, output_file: str, opts: ChangepointsOptions, *,
                         components_file: str | None = None,
                         scores_file: str | None = None, project: Project | None = None) -> ChangepointsResult:
    """Compute the model-free changepoints; write ``<output_dir>/<output_file>.h5``."""
    record = _start("compute-changepoints", opts, input_dir, project, components_file, scores_file)
    compute_changepoints_wrapper(
        input_dir, {**opts.as_dict(), 'pca_file': components_file, 'pca_file_scores': scores_file},
        output_dir, output_file)
    result = ChangepointsResult(changepoints_file=_written(output_dir, output_file))
    stem = str(result.changepoints_file.with_suffix(""))
    _write_record(record, result.changepoints_file, [f"{stem}_dist.png", f"{stem}_dist.pdf"])
    _record(project, "changepoints", result.changepoints_file)
    return result

"""The PCA steps on the extractions they are given: the file layer.

``train_pca``, ``apply_pca`` and ``compute_changepoints`` each take the
extractions to read as (session name, extraction ``.h5``) pairs, in the order to
read them, an output folder and name, and the step's options object, and read no
project file. From a script, pass them directly, such as
``train_pca([("mouse_1", "proc/results_00.h5")], "_pca", "pca", TrainPCAOptions(nworkers=4))``.
The order matters: the changepoints draw one random projection per session in
turn. In a project, ``moseq2_pca.project`` resolves the sessions (those
``moseq2-project.yaml`` lists, less ``exclude``, in the order they were recorded),
the components and scores the project recorded, and the run's record, and calls
these; the command line and the notebook's PCA cells call it.

Each step hands its work function (the ``*_wrapper`` in
``moseq2_pca.helpers.wrappers``) the extractions and the options as a dict, with
the files it reads: ``components_file`` (the trained components;
``<output_dir>/pca.h5`` when None) and, for changepoints, ``scores_file`` (the PC
scores, read for missing-data PCA).

Each ``.h5`` a step writes holds the record of the run (``moseq2_common.records``)
as its ``/provenance`` attribute: the options, every extraction and file read, the
sessions used and left out, and the arrays written, with the figures written
beside it. A step given a record (``record=``, started by its caller) fills it in;
without one, it starts its own, naming the sessions it was given and none left
out. The ``.h5`` and its figures are written under temporary names, the record
stored, and all renamed into place when the step completes
(``moseq2_common.records.atomic_outputs``), so a step stopped partway leaves the
previous output as it was.

When ``train_pca`` or ``apply_pca`` would write over an output that already
exists and its ``overwrite_pca_*`` option is off, the step decides nothing and
prints nothing: it raises ``moseq2_common.project.OverwriteNeeded`` before any
other check (``keep_existing``). Called again with ``replace_existing=True`` it
replaces the output; with ``replace_existing=False`` it keeps it and returns a
result naming no file.

Before any work, a step stops when it is given no extractions, on an extraction
or a components file that does not exist, is empty or cannot be read
(``moseq2_common.records.check_inputs``), and on an extraction whose frames and
timestamps disagree in number (``moseq2_pca.util.check_frame_counts``)
(``check_extractions``).

The components carry the settings they were trained with (``TRAINED_SETTINGS``:
the filters, the height range, FFT and missing-data PCA), as their ``.h5``'s
``/settings`` attribute. ``apply_pca`` and ``compute_changepoints`` prepare each
session's frames with those values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

import h5py

from moseq2_common.bookkeeping import SESSION_ADVICE
from moseq2_common.project import OverwriteNeeded
from moseq2_common.records import InputError, Record, atomic_outputs, check_inputs, start_record, write_h5_record
from moseq2_pca.helpers.data import TRAINED_SETTINGS
from moseq2_pca.helpers.wrappers import apply_pca_wrapper, compute_changepoints_wrapper, train_pca_wrapper
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions
from moseq2_pca.util import check_frame_counts, find_extractions

__all__ = ["COMPONENTS_ADVICE", "NO_PROJECT_CHANGEPOINTS_ADVICE", "NO_PROJECT_COMPONENTS_ADVICE", "TRAINED_SETTINGS",
           "TRAIN_FIGURES", "ApplyPCAResult", "ChangepointsResult", "TrainPCAResult", "apply_pca",
           "check_extractions", "compute_changepoints", "extraction_inputs", "keep_existing", "output_path",
           "train_pca"]

#: The figures train-pca writes beside the components, by the suffix added to their name.
TRAIN_FIGURES = ("_components.png", "_components.pdf", "_scree.png", "_scree.pdf")


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
    #: The changepoints file; None when the computation wrote none.
    changepoints_file: Path | None


#: What to do about the components a PCA step reads, when they are unusable (a session's: ``SESSION_ADVICE``).
COMPONENTS_ADVICE = ("Train PCA again (moseq2-pca train-pca, or the notebook's PCA cell), or give the components "
                     "file it wrote with --pca-file.")
#: Outside a project, a step looks for the components in its output folder.
_NO_PROJECT = ("No project was found (no config.yaml in the working folder, and none given with --config-file), so "
               "{step} looked for the components train-pca writes in its output folder. Run it from the project "
               "folder, or name the project with --config-file /path/to/project/config.yaml, to use the components "
               "its moseq2-project.yaml names; or pass the components file train-pca wrote with --pca-file.")
NO_PROJECT_COMPONENTS_ADVICE = _NO_PROJECT.format(step="apply-pca")
NO_PROJECT_CHANGEPOINTS_ADVICE = _NO_PROJECT.format(step="compute-changepoints")


def output_path(output_dir: str, output_file: str) -> Path:
    """The file a step writes for ``output_dir`` and ``output_file``: ``<output_dir>/<output_file>.h5``."""
    return Path(join(abspath(output_dir), f'{output_file}.h5'))


def keep_existing(output: Path, overwrite: bool, replace_existing: bool | None, *, step: str, option: str) -> bool:
    """Whether a step keeps ``output`` rather than replace it.

    False (go on) when there is no file there, when ``overwrite`` (the step's ``option``) is set, or when
    ``replace_existing`` is True; True (keep it) when ``replace_existing`` is False. When the file exists and
    neither decides, raises ``OverwriteNeeded`` for ``step``.
    """
    if overwrite or not Path(output).exists():
        return False
    if replace_existing is None:
        raise OverwriteNeeded(output, step=step, option=option)
    return not replace_existing


def extraction_inputs(sessions: Sequence[tuple[str, str]]) -> list[dict]:
    """The extractions of ``sessions`` as a record's inputs: each ``.h5`` with its session uuid."""
    h5s, dicts, _ = find_extractions([h5 for _, h5 in sessions])
    return [{"path": h5, "uuid": info["uuid"]} for h5, info in zip(h5s, dicts)]


def check_extractions(step: str, sessions: Sequence[tuple[str, str]], *files: tuple[str, str]) -> None:
    """Stop ``step`` before its work when it was given no extractions, when an extraction of ``sessions`` or one
    of ``files`` (each with what to do about it) is unusable, or when an extraction's frames and timestamps
    disagree in number."""
    if not sessions:
        raise InputError(
            f"{step.replace('-', '_')} was given no extractions to read. Pass each session as a (session name, "
            "path to its extraction .h5) pair."
        )
    h5s = [h5 for _, h5 in sessions]
    check_inputs(step, {**{h5: SESSION_ADVICE for h5 in h5s}, **dict(files)})
    check_frame_counts(h5s)


def _components(components_file, output_dir):
    """The components a step reads: ``components_file``, else ``<output_dir>/pca.h5``."""
    return components_file or join(abspath(output_dir), 'pca.h5')


def _advice(components_file, components_advice, no_project_advice):
    """What to do about unusable components: ``components_advice`` when given; else, as a step run outside a
    project says it, ``COMPONENTS_ADVICE`` for a ``components_file`` given and ``no_project_advice`` for the
    one looked for in the output folder."""
    if components_advice is not None:
        return components_advice
    return COMPONENTS_ADVICE if components_file is not None else no_project_advice


def _settings(opts, output_dir):
    """What a wrapper runs with: ``opts``, with Dask spilling into ``output_dir`` when no ``dask_cache_path``
    is set."""
    settings = opts.as_dict()
    settings['dask_cache_path'] = settings.get('dask_cache_path') or abspath(output_dir)
    return settings


def _start(step, opts, sessions, *files):
    """The record of ``step`` run with no record given: its inputs ``files`` (those not None) and the extractions
    of ``sessions``, noting the sessions given and none left out."""
    record = start_record(step, opts, [*[f for f in files if f], *extraction_inputs(sessions)])
    record.note("sessions", [name for name, _ in sessions])
    record.note("exclude", [])
    return record


def _write_record(record, temporary, written, others=()):
    """Finish ``record`` with the arrays in the ``temporary`` ``.h5`` (written under its final name ``written``)
    and the ``others`` written beside it (final name, temporary name), and store it there."""
    with h5py.File(temporary, "a") as f:
        arrays = {}
        f.visititems(lambda name, node: arrays.setdefault(name, node)
                     if isinstance(node, h5py.Dataset) and not name.startswith("metadata") else None)
        files = [str(written), *[str(final) for final, partial in others if Path(partial).is_file()]]
        write_h5_record(f, record.finish({**arrays, "files": files}))


def _write(record, written, suffixes, work):
    """Run ``work(output_file)``, which writes ``<output_file>.h5`` and ``<output_file><suffix>`` for each of
    ``suffixes`` beside ``written``, under temporary names; store ``record`` in the ``.h5`` and rename them all
    to ``written``'s name when complete (``atomic_outputs``). ``work`` returns whether it wrote the ``.h5``; when
    it did not, nothing is renamed. Returns ``written``, or None."""
    stem = written.with_suffix("")
    with atomic_outputs(stem, (".h5", *suffixes)) as temporary:
        complete = work(temporary.name)
        partial = Path(f"{temporary}.h5")
        if not complete or not partial.is_file():
            for suffix in (".h5", *suffixes):
                Path(f"{temporary}{suffix}").unlink(missing_ok=True)
            return None
        _write_record(record, partial, written, [(f"{stem}{suffix}", f"{temporary}{suffix}") for suffix in suffixes])
    return written


def train_pca(sessions: Sequence[tuple[str, str]], output_dir: str, output_file: str, opts: TrainPCAOptions, *,
              record: Record | None = None, replace_existing: bool | None = None) -> TrainPCAResult:
    """Train PCA on the extractions of ``sessions``; write ``<output_dir>/<output_file>.h5``.

    When that file exists and ``opts.overwrite_pca_train`` is off, ``replace_existing`` decides (None: raise
    ``OverwriteNeeded``). ``record`` is the run's record, started by the caller; without one, the step starts
    its own.
    """
    written = output_path(output_dir, output_file)
    if keep_existing(written, opts.overwrite_pca_train, replace_existing, step="train_pca",
                     option="overwrite_pca_train"):
        return TrainPCAResult(components_file=None)
    check_extractions("train-pca", sessions)
    if record is None:
        record = _start("train-pca", opts, sessions)
    h5s = [h5 for _, h5 in sessions]
    settings = {**_settings(opts, output_dir), 'overwrite_pca_train': True}
    components = _write(record, written, TRAIN_FIGURES, lambda name: bool(
        train_pca_wrapper(h5s, settings, output_dir, name).get('pca_file')))
    return TrainPCAResult(components_file=components)


def apply_pca(sessions: Sequence[tuple[str, str]], output_dir: str, output_file: str, opts: ApplyPCAOptions, *,
              components_file: str | None = None, components_advice: str | None = None,
              record: Record | None = None, replace_existing: bool | None = None) -> ApplyPCAResult:
    """Compute the PC scores of the extractions of ``sessions`` with the components in ``components_file``
    (``<output_dir>/pca.h5`` when None), preparing the frames with the settings the components were trained
    with; write ``<output_dir>/<output_file>.h5``.

    ``components_advice`` is what the message for unusable components says to do (None: as a step run outside a
    project says it). When the output exists and ``opts.overwrite_pca_apply`` is off, ``replace_existing``
    decides (None: raise ``OverwriteNeeded``). ``record`` is the run's record, started by the caller; without
    one, the step starts its own.
    """
    written = output_path(output_dir, output_file)
    if keep_existing(written, opts.overwrite_pca_apply, replace_existing, step="apply_pca",
                     option="overwrite_pca_apply"):
        return ApplyPCAResult(scores_file=None)
    components = _components(components_file, output_dir)
    check_extractions("apply-pca", sessions,
                      (components, _advice(components_file, components_advice, NO_PROJECT_COMPONENTS_ADVICE)))
    if record is None:
        record = _start("apply-pca", opts, sessions, components_file)
    h5s = [h5 for _, h5 in sessions]
    settings = {**_settings(opts, output_dir), 'pca_file': components_file, 'overwrite_pca_apply': True}
    scores = _write(record, written, (), lambda name: apply_pca_wrapper(h5s, settings, output_dir, name)[1])
    return ApplyPCAResult(scores_file=scores)


def compute_changepoints(sessions: Sequence[tuple[str, str]], output_dir: str, output_file: str,
                         opts: ChangepointsOptions, *, components_file: str | None = None,
                         scores_file: str | None = None, components_advice: str | None = None,
                         record: Record | None = None) -> ChangepointsResult:
    """Compute the model-free changepoints of the extractions of ``sessions``, with missing-data PCA when the
    components in ``components_file`` (``<output_dir>/pca.h5`` when None) were trained with it, reading the
    scores in ``scores_file``; write ``<output_dir>/<output_file>.h5``.

    ``components_advice`` is what the message for unusable components says to do (None: as a step run outside a
    project says it). ``record`` is the run's record, started by the caller; without one, the step starts its
    own.
    """
    components = _components(components_file, output_dir)
    check_extractions("compute-changepoints", sessions,
                      (components, _advice(components_file, components_advice, NO_PROJECT_CHANGEPOINTS_ADVICE)))
    if record is None:
        record = _start("compute-changepoints", opts, sessions, components_file, scores_file)
    h5s = [h5 for _, h5 in sessions]
    settings = {**_settings(opts, output_dir), 'pca_file': components_file, 'pca_file_scores': scores_file}

    def work(name):
        compute_changepoints_wrapper(h5s, settings, output_dir, name)
        return True

    changepoints = _write(record, output_path(output_dir, output_file), ("_dist.png", "_dist.pdf"), work)
    return ChangepointsResult(changepoints_file=changepoints)

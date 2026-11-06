"""The modeling steps, one function each.

``learn-model``, ``kappa-scan`` and ``apply-model`` on the command line
(``cli.py``) and the notebook's modeling cells (``gui.py``) resolve their
settings into one options object and call these. A step needs no
``config.yaml``: from a script, pass the files and an options object, such as
``learn_model("_pca/pca_scores.h5", "models/model.p", LearnModelOptions(kappa=1e5))``.

Each step hands its work function (the ``*_wrapper`` in
``moseq2_model.helpers.wrappers``) the options as a dict, with ``index``: the
project's sessions and groups as an index in memory
(``moseq2_common.bookkeeping.session_index``), or None. The index leaves out the
sessions ``config.yaml``'s ``exclude`` lists, and so do the fit and the applied
labels.

Each output carries the record of the run (``moseq2_common.records``): the model
and the kappa scan's script in a ``.yaml`` beside them (``model.p``: ``model.yaml``), an applied model in
its ``.h5``'s ``/provenance`` attribute.

Inside a project (``project``), ``learn_model`` records the model in the
project's bookkeeping file as fitted and current, and ``apply_model`` records
the applied model. A fit run from a kappa scan's job settings file is given no
``project`` and records nothing; its model enters the bookkeeping file when ``get-best-model`` compares
it (``moseq2_viz.steps.get_best_model``).
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import realpath
from pathlib import Path

from moseq2_model.helpers.wrappers import apply_model_wrapper, kappa_scan_fit_models_wrapper, learn_model_wrapper
from moseq2_common.bookkeeping import relative_path, update
from moseq2_common.project import Project
from moseq2_common.records import start_record, write_record
from moseq2_model.options import ApplyModelOptions, KappaScanOptions, LearnModelOptions

@dataclass(frozen=True)
class LearnModelResult:
    model_file: Path
    #: The training log-likelihood figure, written when ``verbose`` is set.
    figure_file: Path | None


@dataclass(frozen=True)
class KappaScanResult:
    #: The script of ``learn-model`` commands.
    script: Path
    #: The script's text.
    commands: str


@dataclass(frozen=True)
class ApplyModelResult:
    applied_file: Path


def _settings(opts, paths):
    return {**paths, **opts.as_dict()}


def _start(step, opts, project, *files):
    """The record of ``step``, its inputs ``files`` and the project's config.yaml."""
    settings = [project.config_file] if project is not None and project.config_file.is_file() else []
    return start_record(step, opts, [*settings, *[f for f in files if f]], project=project)


def _sessions(index):
    """The extraction files the sessions of ``index`` name."""
    if not isinstance(index, dict):
        return []
    entries = index.get("files", [])
    return [entry["path"][0] for entry in (entries.values() if isinstance(entries, dict) else entries)]


def learn_model(input_file: str, dest_file: str, opts: LearnModelOptions, *,
                index: dict | None = None, project: Project | None = None) -> LearnModelResult:
    """Fit an AR-HMM to the PC scores in ``input_file``; write the model to ``dest_file``."""
    record = _start("learn-model", opts, project, input_file, *_sessions(index))
    figure = learn_model_wrapper(input_file, dest_file, _settings(opts, {'index': index}))
    result = LearnModelResult(model_file=Path(realpath(dest_file)), figure_file=Path(figure) if figure else None)
    if isinstance(index, dict) and isinstance(index.get("files"), list):
        record.note("groups", {entry.get("key", entry.get("uuid")): entry.get("group") for entry in index["files"]})
    write_record(result.model_file, record.finish({"files": [str(result.model_file),
                                                             *([str(result.figure_file)] if figure else [])]}))
    if project is not None:
        model = relative_path(project, result.model_file)

        def change(bookkeeping):
            fitted = bookkeeping.models.setdefault("fitted", [])
            if model not in fitted:
                fitted.append(model)
            bookkeeping.models["current"] = model

        update(project, change)
    return result


def kappa_scan(input_file: str, output_dir: str, opts: KappaScanOptions, *, index: dict | None = None,
               learn: LearnModelOptions | None = None, project: Project | None = None) -> KappaScanResult:
    """Write ``opts.out_script``, one ``learn-model`` command per kappa, each saving its model in ``output_dir``.

    Each fit's learn-model settings go to its job settings file beside its model
    (``moseq2_common.slurm.job_settings_file``: ``model-000-1000-job.yaml`` for ``model-000-1000.p``), which
    its command passes to ``--config-file``: the scan's fitting options and, from ``learn``, the learn-model
    settings the scan does not declare (``checkpoint_freq``, ``use_checkpoint``, ``default_group``). The
    file is in config.yaml's shape, its settings in the ``model`` section, with a ``job:`` block naming
    ``project`` and the model. A fit run from it records nothing in ``moseq2-project.yaml``; the scan's
    models enter it when ``get-best-model`` compares them.
    """
    record = _start("kappa-scan", opts, project, input_file)
    commands = kappa_scan_fit_models_wrapper(input_file, _settings(opts, {'index': index}), output_dir,
                                             learn=learn, project=project)
    result = KappaScanResult(script=Path(opts.out_script), commands=commands)
    if result.script.is_file():
        write_record(result.script, record.finish({"files": [str(result.script)]}))
    return result


def apply_model(model_file: str, pc_file: str, dest_file: str, opts: ApplyModelOptions, *,
                index: dict | None = None, project: Project | None = None) -> ApplyModelResult:
    """Label the PC scores in ``pc_file`` with the model in ``model_file``; write them to ``dest_file``."""
    record = _start("apply-model", opts, project, model_file, pc_file, *_sessions(index))
    apply_model_wrapper(model_file, pc_file, dest_file, _settings(opts, {'index': index}))
    result = ApplyModelResult(applied_file=Path(dest_file))
    write_record(result.applied_file, record.finish({"files": [str(result.applied_file)]}))
    if project is not None:
        applied = relative_path(project, result.applied_file)

        def change(bookkeeping):
            if applied not in bookkeeping.applied_models:
                bookkeeping.applied_models.append(applied)

        update(project, change)
    return result

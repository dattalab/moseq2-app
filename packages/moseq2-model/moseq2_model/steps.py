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

Inside a project (``project``), ``learn_model`` records the model in the
project's bookkeeping file as fitted and current, and ``apply_model`` records
the applied model. A fit run from a kappa scan's job settings file is given no
``project`` and records nothing; its model enters the bookkeeping file later.
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import realpath
from pathlib import Path

from moseq2_model.helpers.wrappers import apply_model_wrapper, kappa_scan_fit_models_wrapper, learn_model_wrapper
from moseq2_common.bookkeeping import relative_path, update
from moseq2_common.project import Project
from moseq2_model.options import ApplyModelOptions, KappaScanOptions, LearnModelOptions

#: The file name of each kappa-scan fit's settings, in the scan's output folder beside its model, given the
#: model's number and kappa (``model-000-1000-settings.yaml``). The fit's command runs
#: ``learn-model --config-file`` on it, so the fit gets every learn-model option the scan was given. The file
#: is in config.yaml's shape, its settings in the ``model`` section, with a ``job:`` block naming the project.
KAPPA_SCAN_SETTINGS_NAME = "model-{:03d}-{}-settings.yaml"


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


def learn_model(input_file: str, dest_file: str, opts: LearnModelOptions, *,
                index: dict | None = None, project: Project | None = None) -> LearnModelResult:
    """Fit an AR-HMM to the PC scores in ``input_file``; write the model to ``dest_file``."""
    figure = learn_model_wrapper(input_file, dest_file, _settings(opts, {'index': index}))
    result = LearnModelResult(model_file=Path(realpath(dest_file)), figure_file=Path(figure) if figure else None)
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

    Each fit's learn-model settings go to a ``KAPPA_SCAN_SETTINGS_NAME`` file in ``output_dir``, which its
    command passes to ``--config-file``: the scan's fitting options and, from ``learn``, the learn-model
    settings the scan does not declare (``checkpoint_freq``, ``use_checkpoint``, ``default_group``). Its
    ``job:`` block names ``project``.
    """
    commands = kappa_scan_fit_models_wrapper(input_file, _settings(opts, {'index': index}), output_dir,
                                             KAPPA_SCAN_SETTINGS_NAME, learn=learn, project=project)
    return KappaScanResult(script=Path(opts.out_script), commands=commands)


def apply_model(model_file: str, pc_file: str, dest_file: str, opts: ApplyModelOptions, *,
                index: dict | None = None, project: Project | None = None) -> ApplyModelResult:
    """Label the PC scores in ``pc_file`` with the model in ``model_file``; write them to ``dest_file``."""
    apply_model_wrapper(model_file, pc_file, dest_file, _settings(opts, {'index': index}))
    result = ApplyModelResult(applied_file=Path(dest_file))
    if project is not None:
        applied = relative_path(project, result.applied_file)

        def change(bookkeeping):
            if applied not in bookkeeping.applied_models:
                bookkeeping.applied_models.append(applied)

        update(project, change)
    return result

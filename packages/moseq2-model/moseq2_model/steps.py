"""The modeling steps, one function each.

``learn-model``, ``kappa-scan`` and ``apply-model`` on the command line
(``cli.py``) and the notebook's modeling cells (``gui.py``) resolve their
settings into one options object and call these. A step needs no
``config.yaml``: from a script, pass the files and an options object, such as
``learn_model("_pca/pca_scores.h5", "models/model.p", LearnModelOptions(kappa=1e5))``.

Each step hands its work function (the ``*_wrapper`` in
``moseq2_model.helpers.wrappers``) the options as a dict, with ``index``.
``apply_model`` leaves ``index`` out when it is None, as the notebook's
apply-model always has (L3).

``shim_flat_config`` (R2, removed by PR 4.1) is the notebook's whole
``config.yaml``, passed under the options so the wrappers still see the keys no
modeling command declares: ``select_groups`` (learn-model) and
``default_group`` (apply-model).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os.path import realpath
from pathlib import Path

from moseq2_model.helpers.wrappers import apply_model_wrapper, kappa_scan_fit_models_wrapper, learn_model_wrapper
from moseq2_model.options import ApplyModelOptions, KappaScanOptions, LearnModelOptions

#: The file name of each kappa-scan fit's settings, in the scan's output folder beside its model, given the
#: model's number and kappa (``model-000-1000-settings.yaml``). The fit's command runs
#: ``learn-model --config-file`` on it, so the fit gets every learn-model option the scan was given.
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


def _settings(opts, paths, shim_flat_config):
    return {**(shim_flat_config or {}), **paths, **opts.as_dict()}


def learn_model(input_file: str, dest_file: str, opts: LearnModelOptions, *, index: str | None = None,
                shim_flat_config: Mapping | None = None) -> LearnModelResult:
    """Fit an AR-HMM to the PC scores in ``input_file``; write the model to ``dest_file``."""
    figure = learn_model_wrapper(input_file, dest_file, _settings(opts, {'index': index}, shim_flat_config))
    return LearnModelResult(model_file=Path(realpath(dest_file)), figure_file=Path(figure) if figure else None)


def kappa_scan(input_file: str, output_dir: str, opts: KappaScanOptions, *, index: str | None = None,
               shim_flat_config: Mapping | None = None) -> KappaScanResult:
    """Write ``opts.out_script``, one ``learn-model`` command per kappa, each saving its model in ``output_dir``.

    Each fit's learn-model settings go to a ``KAPPA_SCAN_SETTINGS_NAME`` file in ``output_dir``, which its
    command passes to ``--config-file``: the scan's fitting options, and from ``shim_flat_config`` the
    learn-model options the scan does not declare (``checkpoint_freq``, ``use_checkpoint``,
    ``default_group``, ``verbose``).
    """
    commands = kappa_scan_fit_models_wrapper(input_file, _settings(opts, {'index': index}, shim_flat_config),
                                             output_dir, KAPPA_SCAN_SETTINGS_NAME)
    return KappaScanResult(script=Path(opts.out_script), commands=commands)


def apply_model(model_file: str, pc_file: str, dest_file: str, opts: ApplyModelOptions, *,
                index: str | None = None, shim_flat_config: Mapping | None = None) -> ApplyModelResult:
    """Label the PC scores in ``pc_file`` with the model in ``model_file``; write them to ``dest_file``."""
    paths = {} if index is None else {'index': index}
    apply_model_wrapper(model_file, pc_file, dest_file, _settings(opts, paths, shim_flat_config))
    return ApplyModelResult(applied_file=Path(dest_file))

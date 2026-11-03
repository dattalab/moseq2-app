"""The PCA pipeline steps, one function each.

``train-pca``, ``apply-pca`` and ``compute-changepoints`` on the command line
(``cli.py``) and the notebook's PCA cells (``gui.py``) resolve their settings
into one options object and call these. A step needs no ``config.yaml``: from a
script, pass the paths and an options object, such as
``train_pca("aggregate_results", "_pca", "pca", TrainPCAOptions(nworkers=4))``.

Each step hands its work function (the ``*_wrapper`` in
``moseq2_pca.helpers.wrappers``) the options as a dict.

``shim_flat_config`` (R2, removed by PR 4.1) is the notebook's whole
``config.yaml``, passed under the options so the wrappers still see the keys no
PCA command declares: ``cable_filter_iters`` (training switches to missing-data
PCA when it is above 1), ``data_size`` (apply-pca, compute-changepoints) and
``pca_file`` (compute-changepoints).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

from moseq2_pca.helpers.wrappers import apply_pca_wrapper, compute_changepoints_wrapper, train_pca_wrapper
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions


@dataclass(frozen=True)
class TrainPCAResult:
    #: The components file; None when training wrote none and none was named before.
    components_file: Path | None
    #: The dict the wrapper returned; read only by the front ends' write-back shims (L12, removed by PR 4.3).
    shim_config_data: dict


@dataclass(frozen=True)
class ApplyPCAResult:
    #: The scores file; None when an existing one was kept (overwrite declined).
    scores_file: Path | None
    #: As ``TrainPCAResult.shim_config_data``.
    shim_config_data: dict


@dataclass(frozen=True)
class ChangepointsResult:
    changepoints_file: Path
    #: As ``TrainPCAResult.shim_config_data``.
    shim_config_data: dict


def _settings(opts, shim_flat_config):
    return {**(shim_flat_config or {}), **opts.as_dict()}


def _written(output_dir, output_file):
    """The file a wrapper writes for ``output_dir`` and ``output_file``."""
    return Path(join(abspath(output_dir), f'{output_file}.h5'))


def train_pca(input_dir: str, output_dir: str, output_file: str, opts: TrainPCAOptions, *,
              shim_flat_config: Mapping | None = None) -> TrainPCAResult:
    """Train PCA on the extractions under ``input_dir``; write ``<output_dir>/<output_file>.h5``."""
    config_data = train_pca_wrapper(input_dir, _settings(opts, shim_flat_config), output_dir, output_file)
    pca_file = config_data.get('pca_file')
    return TrainPCAResult(components_file=Path(pca_file) if pca_file else None, shim_config_data=config_data)


def apply_pca(input_dir: str, output_dir: str, output_file: str, opts: ApplyPCAOptions, *,
              shim_flat_config: Mapping | None = None) -> ApplyPCAResult:
    """Compute each extraction's PC scores; write ``<output_dir>/<output_file>.h5``."""
    config_data, success = apply_pca_wrapper(input_dir, _settings(opts, shim_flat_config), output_dir, output_file)
    scores_file = _written(output_dir, output_file) if success else None
    return ApplyPCAResult(scores_file=scores_file, shim_config_data=config_data)


def compute_changepoints(input_dir: str, output_dir: str, output_file: str, opts: ChangepointsOptions, *,
                         shim_flat_config: Mapping | None = None) -> ChangepointsResult:
    """Compute the model-free changepoints; write ``<output_dir>/<output_file>.h5``."""
    config_data = compute_changepoints_wrapper(input_dir, _settings(opts, shim_flat_config), output_dir,
                                               output_file)
    return ChangepointsResult(changepoints_file=_written(output_dir, output_file), shim_config_data=config_data)

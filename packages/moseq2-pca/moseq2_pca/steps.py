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
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

from moseq2_pca.helpers.wrappers import apply_pca_wrapper, compute_changepoints_wrapper, train_pca_wrapper
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions


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


def train_pca(input_dir: str, output_dir: str, output_file: str, opts: TrainPCAOptions) -> TrainPCAResult:
    """Train PCA on the extractions under ``input_dir``; write ``<output_dir>/<output_file>.h5``."""
    config_data = train_pca_wrapper(input_dir, opts.as_dict(), output_dir, output_file)
    pca_file = config_data.get('pca_file')
    return TrainPCAResult(components_file=Path(pca_file) if pca_file else None)


def apply_pca(input_dir: str, output_dir: str, output_file: str, opts: ApplyPCAOptions, *,
              components_file: str | None = None) -> ApplyPCAResult:
    """Compute each extraction's PC scores with the components in ``components_file``; write
    ``<output_dir>/<output_file>.h5``."""
    _, success = apply_pca_wrapper(input_dir, {**opts.as_dict(), 'pca_file': components_file},
                                   output_dir, output_file)
    return ApplyPCAResult(scores_file=_written(output_dir, output_file) if success else None)


def compute_changepoints(input_dir: str, output_dir: str, output_file: str, opts: ChangepointsOptions, *,
                         components_file: str | None = None,
                         scores_file: str | None = None) -> ChangepointsResult:
    """Compute the model-free changepoints; write ``<output_dir>/<output_file>.h5``."""
    compute_changepoints_wrapper(
        input_dir, {**opts.as_dict(), 'pca_file': components_file, 'pca_file_scores': scores_file},
        output_dir, output_file)
    return ChangepointsResult(changepoints_file=_written(output_dir, output_file))

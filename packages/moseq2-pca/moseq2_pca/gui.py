"""
GUI front-end operations for PCA.

Each function reads the ``pca`` section of the project's ``config.yaml``,
resolves the step's settings and calls the step in ``moseq2_pca.steps``.
"""

import warnings

from moseq2_common.options import read_settings
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_pca import steps
from moseq2_pca.cli import recorded_paths, shim_write_back_paths
from moseq2_pca.options import PCA_STAGE, ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions


def shim_index_pca_path(index_file, scores_path):
    """Write the scores' path into the index as ``pca_path`` (L1, removed by PR 4.3).

    As 1.5.0: a missing or unreadable index is reported and left alone.
    """
    try:
        index_params = read_yaml(index_file)
    except OSError:
        index_params = None
    if index_params:
        print(f'Updating index file pca_path: {scores_path}')
        index_params['pca_path'] = scores_path
        write_yaml(index_params, index_file)
    else:
        print('moseq2-index not found, did not update paths')


def _options(cls, config_file):
    return cls.resolve(config=read_settings(config_file, cls, stage_classes=PCA_STAGE))


def train_pca_command(progress_paths, output_dir, output_file):
    """
    Train PCA through Jupyter notebook, and updates config file.

    Args:
    progress_paths (dict): dictionary containing notebook progress paths
    output_dir (str): path to output pca directory
    output_file (str): name of output pca file.

    Returns:
    """
    config_file = progress_paths['config_file']
    opts = _options(TrainPCAOptions, config_file)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        result = steps.train_pca(progress_paths['train_data_dir'], output_dir, output_file, opts)

    shim_write_back_paths(config_file, 'train_pca', result.components_file)


def apply_pca_command(progress_paths, output_file):
    """
    Compute PCA Scores given trained PCA using Jupyter Notebook.

    Args:
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns:
    (str): success string.
    """
    config_file = progress_paths['config_file']
    opts = _options(ApplyPCAOptions, config_file)

    result = steps.apply_pca(progress_paths['train_data_dir'], progress_paths['pca_dirname'], output_file, opts,
                             components_file=recorded_paths(config_file).get('pca_file'))
    success = result.scores_file is not None

    shim_write_back_paths(config_file, 'apply_pca', result.scores_file)

    # if the scores were not recomputed, the path in progress.yaml is still written to the index
    shim_index_pca_path(progress_paths['index_file'], progress_paths.get('scores_path'))

    if success:
        return 'PCA Scores have been successfully computed.'
    else:
        return 'PCA Scores have not been computed'


def compute_changepoints_command(input_dir, progress_paths, output_file):
    """
    Compute Changepoint distribution using Jupyter Notebook.

    Args:
    input_dir (str): path to directory containing training data
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns:
    (str): success string.
    """
    config_file = progress_paths['config_file']
    opts = _options(ChangepointsOptions, config_file)
    recorded = recorded_paths(config_file)

    result = steps.compute_changepoints(input_dir, progress_paths['pca_dirname'], output_file, opts,
                                        components_file=recorded.get('pca_file'),
                                        scores_file=recorded.get('pca_file_scores'))

    shim_write_back_paths(config_file, 'compute_changepoints', result.changepoints_file)

    return 'Model-free syllable changepoints have been successfully computed.'

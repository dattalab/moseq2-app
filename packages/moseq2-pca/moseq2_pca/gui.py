"""
GUI front-end operations for PCA.

Each function reads the project's ``config.yaml``, resolves the step's settings
and calls the step in ``moseq2_pca.steps``.
"""

import warnings

from moseq2_common.options import read_config
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_pca import steps
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions
from .cli import train_pca, apply_pca, compute_changepoints


def shim_notebook_config(command, config):
    """The command's click defaults overlaid with ``config.yaml`` (R2, L12; removed by PR 4.1 and PR 4.3).

    What 1.5.0's notebook functions handed the wrapper under the settings and
    wrote back to ``config.yaml``, paths included.
    """
    return {**{param.name: param.default for param in command.params}, **config}


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
    config = read_config(config_file)
    flat_config = shim_notebook_config(train_pca, config)
    write_yaml(flat_config, config_file)  # shim L12, removed by PR 4.3

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        result = steps.train_pca(progress_paths['train_data_dir'], output_dir, output_file,
                                 TrainPCAOptions.resolve(config=config), shim_flat_config=flat_config)

    write_yaml(result.shim_config_data, config_file)  # shim L12, removed by PR 4.3


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
    config = read_config(config_file)

    result = steps.apply_pca(progress_paths['train_data_dir'], progress_paths['pca_dirname'], output_file,
                             ApplyPCAOptions.resolve(config=config),
                             shim_flat_config=shim_notebook_config(apply_pca, config))
    success = result.scores_file is not None

    if success and result.shim_config_data is not None:
        write_yaml(result.shim_config_data, config_file)  # shim L12, removed by PR 4.3

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
    config = read_config(config_file)

    result = steps.compute_changepoints(input_dir, progress_paths['pca_dirname'], output_file,
                                        ChangepointsOptions.resolve(config=config),
                                        shim_flat_config=shim_notebook_config(compute_changepoints, config))

    write_yaml(result.shim_config_data, config_file)  # shim L12, removed by PR 4.3

    return 'Model-free syllable changepoints have been successfully computed.'

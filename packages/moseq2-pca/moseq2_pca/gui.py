"""
GUI front-end operations for PCA.

"""

import warnings
from moseq2_pca._yaml import write_yaml
from os.path import exists, join
from moseq2_pca.util import read_yaml
from .cli import train_pca, apply_pca, compute_changepoints
from moseq2_pca.helpers.wrappers import train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper


def train_pca_command(progress_paths, output_dir, output_file):
    """
    Train PCA through Jupyter notebook, and updates config file.

    Args:
    progress_paths (dict): dictionary containing notebook progress paths
    output_dir (str): path to output pca directory
    output_file (str): name of output pca file.

    Returns:
    """
    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in train_pca.params if not tmp.required}

    # Get appropriate inputs
    input_dir = progress_paths['train_data_dir']
    config_file = progress_paths['config_file']

    config_data = read_yaml(config_file)
    # merge default params with those in config
    config_data = {**default_params, **config_data}

    write_yaml(config_data, config_file)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        config_data = train_pca_wrapper(input_dir, config_data, output_dir, output_file)

    write_yaml(config_data, config_file)


def apply_pca_command(progress_paths, output_file):
    """
    Compute PCA Scores given trained PCA using Jupyter Notebook.

    Args:
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns:
    (str): success string.
    """
    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in apply_pca.params if not tmp.required}

    # Get proper inputs
    input_dir = progress_paths['train_data_dir']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    output_dir = progress_paths['pca_dirname']

    # outputted scores path
    scores_path = progress_paths.get("scores_path")

    config_data = read_yaml(config_file)
    # merge default params with those in config
    config_data = {**default_params, **config_data}

    config_data, success = apply_pca_wrapper(input_dir, config_data, output_dir, output_file)

    if success:
        if config_data is not None:
            write_yaml(config_data, config_file)
    
    # update the index_file
    # if pc score is not overwritten, the following will ensure the path in progress.yaml will be written to index_file
    # if pc score is overwritten, the new path, updated in line 89 will be written to index_file
    index_params = read_yaml(index_file)
    if index_params:
        print(f'Updating index file pca_path: {scores_path}')
        index_params['pca_path'] = scores_path

        write_yaml(index_params, index_file)
    else:
        print('moseq2-index not found, did not update paths')

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
    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in compute_changepoints.params
                      if not tmp.required}

    config_file = progress_paths['config_file']
    output_dir = progress_paths['pca_dirname']

    config_data = read_yaml(config_file)
    # merge default params with those in config
    config_data = {**default_params, **config_data}

    config_data = compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file)

    write_yaml(config_data, config_file)

    return 'Model-free syllable changepoints have been successfully computed.'

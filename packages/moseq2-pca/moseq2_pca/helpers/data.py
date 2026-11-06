"""
Helper functions for reading and loading PCA data.
"""

import io

import h5py
from moseq2_common.yaml import dump_block_yaml, read_yaml
from moseq2_common.records import StepInputError
from os.path import join, exists
from moseq2_common.frames import select_strel


def get_pca_paths(config_data, output_dir):
    """
    Helper function for changepoints_wrapper to perform data-path existence checks.
    Returns paths to saved pre-trained PCA components and PCA Scores files.

    Args:
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data

    Returns:
    config_data (dict): updated config_data dict with the pc component and pc score paths
    pca_file_components (str): path to trained pca file
    pca_file_scores (str): path to pca_scores file
    """

    # Check if there is PCA file from config_data
    if config_data.get('pca_file', None) is not None:
        pca_file = config_data['pca_file']
    else:
        # Assume PCA file is in output_dir
        pca_file = join(output_dir, 'pca.h5')
        config_data['pca_file'] = pca_file

    if not exists(pca_file):
        raise IOError(f'Could not find PCA components file {pca_file}')

    # Get path to PCA Scores
    pca_file_scores = config_data.get('pca_file_scores', join(output_dir, 'pca_scores.h5'))
    config_data['pca_file_scores'] = pca_file_scores

    return config_data, pca_file, pca_file_scores

def load_pcs_for_cp(pca_file, config_data):
    """
    Load computed Principal Components for Model-free Changepoint Analysis.

    Args:
    pca_file (str): path to pca h5 file to read PCs
    config_data (dict): config parameters

    Returns:
    pca_components (numpy.ndarray): the components
    changepoint_params (dict): dict of relevant changepoint parameters
    missing_data (bool): whether the components were trained with missing-data PCA, so the masks are used
    mask_params (dict): Mask parameters to use when computing CPs
    """

    print(f'Loading PCs from {pca_file}')
    with h5py.File(pca_file, 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # missing-data PCA is a training setting the components carry
    trained = read_trained_settings(pca_file)
    missing_data = trained['missing_data']
    mask_params = None
    if missing_data:
        print('Detected missing data...')
        mask_params = {'mask_height_threshold': trained['mask_height_threshold'],
                       'mask_threshold': trained['mask_threshold']}
        if not exists(config_data['pca_file_scores']):
            raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    # Pack changepoint parameters
    changepoint_params = {
        'k': config_data['klags'],
        'sigma': config_data['sigma'],
        'peak_height': config_data['threshold'],
        'peak_neighbors': config_data['neighbors'],
        'rps': config_data['dims']
    }

    return pca_components, changepoint_params, missing_data, mask_params


#: The ``train-pca`` settings a components file carries (``write_trained_settings``): the filters, the height
#: range, FFT, and missing-data PCA with its masks. ``apply-pca`` and ``compute-changepoints`` prepare each
#: session's frames with these values, the ones the components were trained on.
TRAINED_SETTINGS = ('gaussfilter_space', 'gaussfilter_time', 'medfilter_space', 'medfilter_time',
                    'tailfilter_shape', 'tailfilter_size', 'min_height', 'max_height', 'use_fft',
                    'missing_data', 'mask_threshold', 'mask_height_threshold')

class TrainedSettingsError(StepInputError, ValueError):
    """The PCA components cannot be used with these settings: ``config.yaml`` sets one they were trained with to
    another value, or the components do not say what they were trained with."""


#: The attribute of the components ``.h5`` holding ``TRAINED_SETTINGS``, as one YAML text block.
SETTINGS_ATTRIBUTE = 'settings'


def write_trained_settings(h5, config_data):
    """Store the ``TRAINED_SETTINGS`` of ``config_data`` in the open components file ``h5``."""
    h5.attrs[SETTINGS_ATTRIBUTE] = dump_block_yaml({key: config_data[key] for key in TRAINED_SETTINGS})


def read_trained_settings(pca_file):
    """
    The settings the components in ``pca_file`` were trained with (``TRAINED_SETTINGS``).

    Args:
    pca_file (str): path to the components file.

    Returns:
    (dict): each setting's trained value, as YAML holds it (lists for tuples).
    """
    with h5py.File(pca_file, 'r') as f:
        text = f.attrs.get(SETTINGS_ATTRIBUTE)
    if text is None:
        raise TrainedSettingsError(
            f'The PCA components in {pca_file} do not say which settings they were trained with (filters, '
            'height range, missing-data PCA), so their scores and changepoints cannot be computed the same way. '
            'They were trained by MoSeq2 1.5.0 or earlier. Train PCA again: `moseq2-pca train-pca`, or the '
            'notebook\'s PCA training cell.'
        )
    stored = read_yaml(io.StringIO(text.decode() if isinstance(text, bytes) else text)) or {}
    missing = [key for key in TRAINED_SETTINGS if key not in stored]
    if missing:
        raise TrainedSettingsError(
            f'The PCA components in {pca_file} do not say which {", ".join(missing)} they were trained with. '
            'Train PCA again: `moseq2-pca train-pca`, or the notebook\'s PCA training cell.'
        )
    return {key: stored[key] for key in TRAINED_SETTINGS}


def trained_cleaning(pca_file):
    """
    How the components in ``pca_file`` were trained to see frames, as the scoring functions take it.

    Args:
    pca_file (str): path to the components file.

    Returns:
    use_fft (bool): indicates whether to use FFT
    clean_params (dict): dict of image filtering parameters
    mask_params (dict): dict of mask parameters and the height range
    missing_data (bool): indicates whether to use mask_params
    """
    trained = read_trained_settings(pca_file)

    use_fft = trained['use_fft']
    missing_data = trained['missing_data']
    if use_fft:
        print('Will use FFT...')
    if missing_data:
        print('Detected missing data...')

    clean_params = {
        'gaussfilter_space': trained['gaussfilter_space'],
        'gaussfilter_time': trained['gaussfilter_time'],
        'tailfilter': select_strel(trained['tailfilter_shape'], tuple(trained['tailfilter_size'])),
        'medfilter_time': trained['medfilter_time'],
        'medfilter_space': trained['medfilter_space'],
    }

    mask_params = {
        'mask_height_threshold': trained['mask_height_threshold'],
        'mask_threshold': trained['mask_threshold'],
        'min_height': trained['min_height'],
        'max_height': trained['max_height']
    }

    return use_fft, clean_params, mask_params, missing_data

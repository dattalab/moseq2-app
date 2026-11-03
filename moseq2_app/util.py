"""
General utility functions.
"""
import pandas as pd
from moseq2_common.yaml import read_yaml, write_yaml as _write_yaml
from copy import deepcopy
import os
import warnings
from os.path import abspath, basename, dirname, join, exists, splitext
from os import mkdir
from glob import glob
from shutil import copy2
from collections import defaultdict
from contextlib import contextmanager
from IPython.display import clear_output
from moseq2_app.gui.progress import update_progress
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_extract.util import check_filter_sizes, recursive_find_h5s
from moseq2_extract.recordings import find_recordings
from moseq2_viz.model.util import compute_behavioral_statistics


class ClearOutputMixin:
    """
    The Clear Output button's handler, shared by the notebook tools that show one.
    """

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()


def read_and_clean_config(config_file):
    """read config files and reset incorrect parameters

    Args:
        config_file (dict): path to the config file

    Returns:
        config_data (dict): dictionary of config data.
    """
    config_data = read_yaml(config_file)
    config_data = check_filter_sizes(config_data)
    config_data['threads'] = max(1, config_data.get('threads', 8))

    # TODO: see if this is necessary
    # set defaults if the keys don't exist
    config_data = {
        'bg_roi_erode': (1, 1),
        'bg_roi_dilate': (1, 1),
        **config_data
    }

    return config_data


def merge_labels_with_scalars(sorted_index, model_path):
    """
    Compute all the syllable statistics to plot, including syllable scalars.

    Args:
    sorted_index (dict): Sorted dict of modeled sessions
    model_fit (dict): Trained AR-HMM results dict
    model_path (str): Respective path to the AR-HMM model in use.
    max_sylls (int): Maximum number of syllables to include

    Returns:
    df (pd.DataFrame): Dataframe containing all of the mean syllable statistics
    scalar_df (pd.DataFrame): Dataframe containing the frame-by-frame scalar and label data
    """

    # Load scalar Dataframe to compute syllable speeds
    scalar_df = scalars_to_dataframe(sorted_index, model_path=model_path)

    df = compute_behavioral_statistics(scalar_df, count='usage',
                                       groupby=['group', 'uuid', 'SessionName', 'SubjectName'])

    return df, scalar_df

def index_to_dataframe(index_path):
    """
    Read the index file into a dictionary and converts it into an editable DataFrame.

    Args:
    index_path (str): Path to index file

    Returns:
    index_data (dict): Dict object containing all parsed index file contents
    df (pd.DataFrame): Formatted dict in DataFrame form including each session's metadata
    """

    index_data = read_yaml(index_path)

    files = index_data['files']
    meta = [f['metadata'] for f in files]

    meta_df = pd.DataFrame(meta)
    tmp_df = pd.DataFrame(files)

    df = pd.concat([meta_df, tmp_df], axis=1)

    def apply_filename(n):
        return basename(n[0])

    df['filename'] = df['path'].apply(apply_filename)

    return index_data, df

class bcolors:
    """
    color UNICODE values used to color printed output.
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def uuid_lookup(target_uuid, uuid_dict_source):
    """
    Look up session infomtion with full/partial uuid. Helper function for users to look up uuid after running interactive_scalar_summary

    Args:
    target_uuid (str): full or partial uuid the user wants to look up.
    uuid_dict (dict): dictionary from interactive_scalar_summary widget that has all the session information 
    """
    
    # deep copy the input dictionary
    uuid_dict = deepcopy(uuid_dict_source)

    for uuid, info in uuid_dict.items():
        if target_uuid in uuid:
            print('UUID:', uuid)
            # Printed a field at a time rather than as a dict, so that a path
            # arrives as a path. Rendering the dict quotes every value and
            # escapes each separator, which on Windows turns a path the user
            # is being handed in order to go and open it into one they have to
            # edit first.
            metadata = info['metadata']
            metadata['h5Path'], metadata['yamlPath'] = info['path']
            for field in ('SessionName', 'SubjectName', 'StartTime', 'h5Path', 'yamlPath'):
                print(f'  {field}: {metadata[field]}')

def setup_model_folders(progress_paths):
    """Create model-specific folders

    Args:
    progress_paths (dict): dictionary of notebook progress paths.

    Returns:
    model_dict (dict): dictionary for model specific paths such as model_session_path, model_path, syll_info and crowd_dir
    """
    # find all the models in the model master path
    models = glob(join(progress_paths['base_model_path'], '*.p'))
    
    # initialize model dictionary
    model_dict = defaultdict(dict)

    for model in models:
        model_dir = splitext(model)[0]
        # get the model file name to use as 
        model = basename(model)
        # Check if the model directory already exists
        if not exists(model_dir):
            print('Creating model folder for', model)
            # make a model-specific folder
            mkdir(model_dir)
        
        # check if the model is copied to the model-specific folder
        if not exists(join(model_dir, model)):
            copy2(join(progress_paths['base_model_path'], model), model_dir)
        
        model_dict[model]['model_session_path'] = model_dir
        model_dict[model]['model_path'] = join(model_dir, model)
    return dict(model_dict)  # remove defaultdict class

def update_model_paths(desired_model, model_dict, progress_filepath):
    """Update relevant model paths in progress.yaml when specific model is chosen

    Args:
    desired_model (str): file name of the desired specific model
    model_dict (dict): dictionary for model specific paths such as model_session_path, model_path, syll_info and crowd_dir
    progress_filepath (str): path to progress.yaml

    Returns:
    progress_paths (dict): dictionary of paths in the analysis
    """

    assert desired_model in model_dict, '{} not found in model_dict. Make sure desired_model is one of the keys in model_dict. \nPossible keys: \n{}'.format(desired_model, "\n".join(map(str, model_dict)))

    # update model_session_path and model_path
    for key in ['model_session_path', 'model_path']:
        progress_paths = update_progress(progress_filepath, key, model_dict[desired_model].get(key))
    progress_paths = update_progress(progress_filepath, 'plot_path', join(model_dict[desired_model]['model_session_path'], 'plots/'))
    progress_paths = update_progress(progress_filepath, 'crowd_dir', join(model_dict[desired_model]['model_session_path'], 'crowd_movies/'))
    progress_paths = update_progress(progress_filepath, 'syll_info', join(model_dict[desired_model]['model_session_path'], 'syll_info.yaml'))

    return progress_paths


@contextmanager
def update_config(path: str) -> dict:
    """update config.yaml with new parameters used.

    Args:
        path (str): path to config file.
    """
    config = read_yaml(path)
    try:
        yield config
    finally:
        _write_yaml(config, path)


#: The ``camera_type`` values under which the PCA cell decides the camera from
#: the recordings: the Kinect v2 (``k2``, ``kinect``) and ``auto``. Notebooks from
#: 1.4.5 and earlier left ``k2`` or ``auto`` in an Azure or Femto project's config.
_CAMERA_FROM_RECORDINGS = ('k2', 'kinect', 'auto')


def _recording_kinds(train_data_dir, project_dir):
    """Count the sessions in ``train_data_dir`` by the recording each was extracted from.

    An aggregated session shares its uuid with the extraction it was copied from,
    which sits in an output folder at or below its recording's folder. The
    recording is the one ``find_recordings`` lists in the nearest folder above
    that extraction.

    Returns:
    (dict): ``{'mkv': n, 'other': n, 'not found': n}``.
    """
    _, trained, _ = recursive_find_h5s(train_data_dir)
    train_root = abspath(train_data_dir)
    project_root = abspath(project_dir)

    extractions = {}
    for h5, results in zip(*recursive_find_h5s(project_dir)[:2]):
        if not abspath(h5).startswith(train_root + os.sep):
            extractions.setdefault(results.get('uuid'), abspath(h5))

    recordings = {}
    for recording in find_recordings(project_dir).recordings:
        recordings.setdefault(dirname(abspath(recording)), []).append(recording)

    kinds = {'mkv': 0, 'other': 0, 'not found': 0}
    for results in trained:
        folder = extractions.get(results.get('uuid'))
        found = None
        while folder is not None and found is None:
            folder = dirname(folder)
            found = recordings.get(folder)
            if folder == project_root or dirname(folder) == folder:
                break
        if not found:
            kinds['not found'] += 1
        elif any(recording.endswith('.mkv') for recording in found):
            kinds['mkv'] += 1
        else:
            kinds['other'] += 1
    return kinds


def camera_for_pca(camera_type, train_data_dir, project_dir):
    """The camera whose filters the PCA cell applies: ``'azure'`` or ``camera_type``.

    Only Azure Kinect and Orbbec Femto cameras write ``.mkv``. When ``camera_type``
    is ``k2``, ``kinect`` or ``auto`` and every session in ``train_data_dir`` was
    extracted from an ``.mkv``, the camera is ``'azure'``, and a line says so. When
    some sessions are ``.mkv`` and others are not, or their recordings cannot be
    found, ``camera_type`` is kept and a warning gives the counts. Any other
    ``camera_type`` is returned as it is.

    Args:
    camera_type (str): ``camera_type`` from config.yaml.
    train_data_dir (str): the folder PCA trains on (``aggregate_results/``).
    project_dir (str): the project folder holding the recordings and their extractions.

    Returns:
    (str): ``'azure'`` or ``camera_type``.
    """
    if camera_type not in _CAMERA_FROM_RECORDINGS:
        return camera_type

    kinds = _recording_kinds(train_data_dir, project_dir)
    n_mkv = kinds['mkv']
    if n_mkv == 0:
        return camera_type

    if n_mkv == sum(kinds.values()):
        sessions = (f'the 1 session in {train_data_dir} was recorded as an .mkv' if n_mkv == 1
                    else f'all {n_mkv} sessions in {train_data_dir} were recorded as .mkv files')
        print('PCA uses the Azure Kinect and Orbbec Femto filters set in this cell '
              f'(gaussfilter_space, tailfilter_size): config.yaml says camera_type: {camera_type}, '
              f'but {sessions}, which only those cameras write. Set camera_type: azure in config.yaml.')
        return 'azure'

    parts = [f"{n_mkv} {'was' if n_mkv == 1 else 'were'} recorded as .mkv (an Azure Kinect or Orbbec Femto)"]
    if kinds['other']:
        parts.append(f"{kinds['other']} as another file type (.dat, .avi or .tar.gz)")
    if kinds['not found']:
        parts.append(f"for {kinds['not found']} the recording could not be found under {project_dir}")
    counts = parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]
    warnings.warn(
        f'Not every session in {train_data_dir} was recorded as an .mkv: {counts}. PCA uses the '
        f"filters for config.yaml's camera_type: {camera_type}, which suit the Kinect v2. If every "
        'session is from an Azure Kinect or Orbbec Femto, set camera_type: azure in config.yaml '
        'and run this cell again.'
    )
    return camera_type

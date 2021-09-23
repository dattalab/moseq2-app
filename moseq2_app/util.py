'''

General utility functions.

'''
import pandas as pd
from os.path import basename, join, exists
from os import listdir, mkdir
from shutil import copy2
from collections import defaultdict
from moseq2_app.gui.progress import update_progress
from moseq2_viz.util import read_yaml
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.model.util import compute_behavioral_statistics

def merge_labels_with_scalars(sorted_index, model_path):
    '''
    Computes all the syllable statistics to plot, including syllable scalars.

    Parameters
    ----------
    sorted_index (dict): Sorted dict of modeled sessions
    model_fit (dict): Trained ARHMM results dict
    model_path (str): Respective path to the ARHMM model in use.
    max_sylls (int): Maximum number of syllables to include

    Returns
    -------
    df (pd.DataFrame): Dataframe containing all of the mean syllable statistics
    scalar_df (pd.DataFrame): Dataframe containing the frame-by-frame scalar and label data
    '''

    # Load scalar Dataframe to compute syllable speeds
    scalar_df = scalars_to_dataframe(sorted_index, model_path=model_path)

    df = compute_behavioral_statistics(scalar_df, count='usage',
                                       groupby=['group', 'uuid', 'SessionName', 'SubjectName'])

    return df, scalar_df

def index_to_dataframe(index_path):
    '''
    Reads the index file into a dictionary and converts it into an editable DataFrame.

    Parameters
    ----------
    index_path (str): Path to index file

    Returns
    -------
    index_data (dict): Dict object containing all parsed index file contents
    df (pd.DataFrame): Formatted dict in DataFrame form including each session's metadata
    '''

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
    '''
    Class containing color UNICODE values used to color printed output.
    '''

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def setup_model_folders(progress_paths):
    """Create model-specific folders

    Parameters
    ----------
    progress_paths (dict): dictionary of notebook progress paths.

    Returns
    -------
    model_dict (dict): dictionary for model specific paths such as model_session_path, model_path, syll_info, syll_info_df and crowd_dir
    """
    # find all the models in the model master path
    models = [file for file in listdir(progress_paths['model_master_path']) if file.endswith('.p')]
    
    # initialize model dictionary
    model_dict = defaultdict(dict)

    for model in models:
        model_dir = join(progress_paths['model_master_path'], model.split('.')[0])
        # enusre previous progress are not wiped
        if exists(model_dir):
            if exists(join(model_dir, model)):
                pass
            else:
                copy2(join(progress_paths['model_master_path'], model), model_dir)
        else:
            mkdir(model_dir)
            copy2(join(progress_paths['model_master_path'], model), model_dir)
            print('Creating the model folder...')
        model_dict[model]['model_session_path'] = model_dir
        model_dict[model]['model_path'] = join(model_dir, model)
    return model_dict

def update_model_paths(desired_model, model_dict, progress_filepath):
    """helper function to update relevant model paths in progress.yaml when specific model is chosen

    Parameters
    ----------
    desired_model (str): file name of the desired specific model
    model_dict (dict): dictionary for model specific paths such as model_session_path, model_path, syll_info, syll_info_df and crowd_dir
    progress_filepath (str): path to progress.yaml

    Returns
    -------
    [type]
        [description]
    """    
    progress_paths = update_progress(progress_filepath, 'model_session_path', model_dict[desired_model].get('model_session_path'))
    progress_paths = update_progress(progress_filepath, 'model_path', model_dict[desired_model].get('model_path'))

    # remove previously stored model-specific paths
    progress_paths = update_progress(progress_filepath, 'crowd_dir', '')
    progress_paths = update_progress(progress_filepath, 'syll_info', '')
    progress_paths = update_progress(progress_filepath, 'df_info_path', '')
    return progress_paths
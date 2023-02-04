"""
This module handles all jupyter notebook progress related functionalities.
"""

import os
import uuid
import json
import pickle
from glob import glob
from time import sleep
import ruamel.yaml as yaml
from operator import add
from toolz import compose
from tqdm.auto import tqdm
from functools import reduce
from datetime import datetime
from moseq2_viz.util import read_yaml
from os.path import dirname, basename, exists, join, abspath
from moseq2_extract.helpers.data import check_completion_status


def generate_missing_metadata(sess_dir, sess_name):
    """
    Generate default metadata.json file for session that does not already include one.

    Args:
    sess_dir (str): Path to directory to create metadata.json file in.
    sess_name (str): Name of the directory to set the metadata SessionName.
    """

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': f'{basename(sess_dir)}', f'SessionName': f'{sess_name}',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    with open(join(sess_dir, 'metadata.json'), 'w') as fp:
        json.dump(sample_meta, fp)


def _is_unextracted(folder):
    """
    check if ny sessions in the folder is unextracted.

    Args:
        folder (str): path to depth recording
    
    Return:
        (bool): whether the session is extracted
    """
    if not exists(join(folder, 'proc')):
        return True
    elif not exists(join(folder, 'proc', 'results_00.yaml')):
        return True
    # if results.yaml exists, then check if extraction has successfully completed
    results_dict = read_yaml(join(folder, 'proc', 'results_00.yaml'))
    return not results_dict.get('complete', False)


def _has_metadata(folder):
    """
    check if all sessions have metadata.

    Args:
        folder (str): path to depth recording

    Returns:
        (bool): whether there is metadata
    """
    return exists(join(folder, 'metadata.json'))


def get_sessions(data_dir, skip_extracted=True, extensions=('dat', 'mkv', 'avi', 'tar.gz')):
    """get sessions to extract

    Args:
        data_dir (str): path to project directory
        skip_extracted (bool, optional): bool flag for skipping extracted sessions. Defaults to True.
        extensions (tuple, optional): extensions to look for depth recordings. Defaults to ('dat', 'mkv', 'avi', 'tar.gz').

    Returns:
        _type_: _description_
    """
    # look for files in subfolders
    files = [glob(join(data_dir, '**', f'*.{ext}'), recursive=True) for ext in extensions]
    # concatenate all files of different extensions
    files = sorted(reduce(add, files))

    # remove any folder that doesn't have a metadata.json file
    files = list(filter(compose(_has_metadata, dirname), files))

    if skip_extracted:
        # filter any folders that have been extracted
        files = list(filter(compose(_is_unextracted, dirname), files))

    return files


def get_session_paths(data_dir, extracted=False, flipped=False, exts=['dat', 'mkv', 'avi']):
    """
    Find all depth recording sessions and their paths (with given extensions) to work on given base directory.

    Args:
    data_dir (str): path to directory containing all session folders.
    exts (list): list of depth file extensions to search for.
    flipped (bool): indicates whether to show corrected flip videos
    extracted (bool): indicates to return paths to extracted sessions only.

    Returns:
    path_dict (dict): session directory name keys pair with their respective absolute paths.
    """

    if extracted:
        path = '*/proc/*.'
        if flipped:
            path = '*/proc/*_flipped.'
        exts = ['mp4']
    else:
        path = '*/*.'

    sessions = []

    # Get list of sessions ending in the given extensions
    for ext in exts:
        if len(data_dir) == 0:
            data_dir = os.getcwd()
            if flipped:
                files = sorted(glob(path + ext))
            else:
                files = [f for f in sorted(glob(path + ext)) if 'flipped' not in f]
            sessions += files
        else:
            data_dir = data_dir.strip()
            if os.path.isdir(data_dir):
                if flipped:
                    files = sorted(glob(os.path.join(data_dir, path + ext)))
                else:
                    files = [f for f in sorted(glob(os.path.join(data_dir, path + ext))) if 'flipped' not in f]
                sessions += files
            else:
                print('directory not found, try again.')

    if len(sessions) == 0:
        if extracted:
            sessions = sorted(glob(join(data_dir, '*.mp4')))
        else:
            for ext in exts:
                sessions += sorted(glob(join(data_dir, f'*.{ext}')))

    if extracted:
        names = [dirname(sess).split('/')[-2] for sess in sessions]
        if len(set(names)) == len(sessions):
            path_dict = {n: p for n, p in zip(names, sessions)}
        else:
            path_dict = {basename(p): p for p in sessions}
    else:
        for sess in sessions:
            # get path to session directory
            sess_dir = dirname(sess)
            sess_name = basename(sess_dir)
            if 'metadata.json' not in os.listdir(sess_dir):
                # Generate metadata.json file if it's missing
                generate_missing_metadata(sess_dir, sess_name)

        # Create path dictionary
        names = [basename(dirname(sess)) for sess in sessions]
        if len(set(names)) == len(sessions):
            path_dict = {n: p for n, p in zip(names, sessions)}
        else:
            path_dict = {basename(p): p for p in sessions}

    return path_dict

def update_progress(progress_file, varK, varV):
    """
    Update progress file with new notebook variable

    Args:
    progress_file (str): path to progress file
    varK (str): key in progress file to update
    varV (str): updated value to write

    Returns:
    progress (dict): Loaded path dict from the progress yaml file.
    """

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    progress = read_yaml(progress_file)

    if isinstance(varV, str):
        old_value = progress.get(varK, '') # get previous variable to print        

        if old_value == varV:
            print('Variables are the same. No update necessary.')
            return progress

        progress[varK] = varV

        # update snapshot variable
        progress['snapshot'] = str(uuid.uuid4())

        with open(progress_file, 'w') as f:
            yml.dump(progress, f)

        print(f'Successfully updated progress file with {varK} -> {varV}')
    else:
        print('Entered path is invalid.')
        print('Ensure you are updating the progress file with string paths only.')

    return progress

def find_progress(base_progress):
    """
    Search for paths to all existing MosSeq2-Notebook dependencies and updates the progress paths dictionary.

    Args:
    base_progress (dict): base dictionary of progress variables

    Returns:
    base_progress (dict): updated dictionary of progress variables
    """

    base_dir = base_progress['base_dir']
    yamls = glob(join(base_dir, '*.yaml'))

    if join(base_dir, 'config.yaml') in yamls:
        base_progress['config_file'] = join(base_dir, 'config.yaml')

    if join(base_dir, 'session_config.yaml') in yamls:
        base_progress['session_config'] = join(base_dir, 'session_config.yaml')

    if join(base_dir, 'moseq2-index.yaml') in yamls:
        base_progress['index_file'] = join(base_dir, 'moseq2-index.yaml')

    if exists(join(base_dir, 'aggregate_results/')):
        base_progress['train_data_dir'] = join(base_dir, 'aggregate_results/')
    
    # initialize param
    pca_score = None
    changepoint = None
    # Read config.yaml to get the pca related paths
    if exists(base_progress['config_file'] ):
        config_data = read_yaml(base_progress['config_file'])
        
        pca_score = config_data.get('pca_file_scores')
        changepoint = config_data.get('changepoint_file')
    
    # if pca_score is in config.yaml and the file exists, use that in the progress dictionary
    if pca_score and exists(pca_score):
        base_progress['pca_dirname'] = abspath(dirname(pca_score))
        base_progress['scores_filename'] = basename(pca_score)
        base_progress['scores_path'] = abspath(pca_score)
    # use default
    elif exists(join(base_dir, '_pca/')):
        base_progress['pca_dirname'] = join(base_dir, '_pca/')
        base_progress['scores_filename'] = 'pca_scores'
        if exists(join(base_progress['pca_dirname'], base_progress['scores_filename'] +'.h5')):
            base_progress['scores_path'] = join(base_dir, '_pca/', 'pca_scores.h5')
    else:
        print('Unable to find PC score file. Either:\n    1) run the pca step, or if you did\n    2) manually add PCA paths using the update_progress function')
    
    # Add pc score to moseq2-index.yaml file if it is empty because it is run from cli
    if base_progress.get('index_file') and exists(base_progress.get('index_file')):
        index_params = read_yaml(base_progress.get('index_file'))
        if base_progress.get('scores_path') and exists(base_progress.get('scores_path')):
            index_params['pca_path'] = base_progress.get('scores_path')
            with open(base_progress.get('index_file'), 'w') as f:
                yaml.safe_dump(index_params, f)
        else:
            print('Please ensure "pca_path" in moseq2-index.yaml is the path to pc_score h5 file before running interactive model analysis')
    
    # if changepoint is in config.yaml and the file exists, use that in the progress dictionary
    if changepoint and exists(changepoint):
        base_progress['changepoints_path'] = changepoint
    # use default
    elif exists(join(base_progress['pca_dirname'], 'changepoints.h5')):
        base_progress['changepoints_path'] = join(base_progress['pca_dirname'], 'changepoints.h5')
    else:
        print('Unable to find changepoint file. Either:\n    1) run the pca step, or if you did\n    2) manually add PCA paths using the update_progress function')
             
    models = glob(join(base_dir, '**/*.p'), recursive=True)

    if len(models) > 1:
        models = sorted(models, key=os.path.getmtime)
        print(f'More than 1 model found. Setting model path to latest generated model: {models[0]}')
    # avoid models not found
    if len(models) > 0:
        base_progress['model_path'] = models[0]
        base_progress['model_session_path'] = dirname(models[0])
        base_progress['base_model_path'] = dirname(models[0])
        if exists(join(dirname(models[0]), 'syll_info.yaml')):
            base_progress['syll_info'] = join(dirname(models[0]), 'syll_info.yaml')
        if exists(join(dirname(models[0]), 'crowd_movies/')):
            base_progress['crowd_dir'] = join(dirname(models[0]), 'crowd_movies/')
    return base_progress

def generate_intital_progressfile(filename='progress.yaml'):
    """
    Generate a progress YAML file with the scanned parameter paths.

    Args:
    filename  (str): path to file to write progress YAML to

    Returns:
    base_progress_vars (dict): Loaded/Found progress variables
    """

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    base_dir = dirname(filename)

    print(f'Generating progress path at: {filename}')

    # Create basic progress file
    base_progress_vars = {'base_dir': abspath(base_dir),
                          'config_file': '',
                          'session_config': '',
                          'index_file': '',
                          'train_data_dir': '',
                          'pca_dirname': '',
                          'scores_filename': '',
                          'scores_path': '',
                          'changepoints_path': '',
                          'model_path': '',
                          'crowd_dir': '',
                          'syll_info': '',
                          'plot_path': join(base_dir, 'plots/'),
                          'snapshot': str(uuid.uuid4())}

    # Find progress in given base directory
    base_progress_vars = find_progress(base_progress_vars)

    with open(filename, 'w') as f:
        yml.dump(base_progress_vars, f)

    curr_id = base_progress_vars['snapshot']

    return base_progress_vars

def load_progress(progress_file):
    """
    Load progress file variables

    Args:
    progress_file (str): path to progress file.

    Returns:
    progress_vars (dict): dictionary of loaded progress variables
    """

    if exists(progress_file):
        print('Updating notebook variables...')
        progress_vars = read_yaml(progress_file)
    else:
        print('Progress file not found. To generate a new one, set restore_progress_vars(progress_file, init=True)')
        progress_vars = None

    return progress_vars

def restore_progress_vars(progress_file=abspath('./progress.yaml'), init=False, overwrite=False):
    """
    Restore all saved progress variables to Jupyter Notebook.

    Args:
    progress_file (str): path to progress file

    Returns:
    vars (dict): All progress file variables
    """

    # overwrite the progress file is overwrite is True
    if overwrite:
        print('Overwriting progress file with initial progress.')
        progress_vars = generate_intital_progressfile(progress_file)
    else:
        if init:
            # restore progress file if it exists
            if exists(progress_file):
                progress_vars = load_progress(progress_file)
            # generate new progress file if it doesn't exists
            else:
                progress_vars = generate_intital_progressfile(progress_file)
        # restore progress file if it is not init
        else:
            progress_vars = load_progress(progress_file)

    return progress_vars

def get_pca_progress(progress_vars, pca_progress):
    """
    Update the PCA progress dict variables and prints the names of the missing keys.

    Args:
    progress_vars (dict): progress dict including the relevant PCA paths
    pca_progress (dict): PCA progress boolean dict used to display progress bar

    Returns:
    pca_progress (dict): Updated PCA progress boolean dict.
    """

    # Get PCA Progress
    for key in pca_progress:
        if progress_vars.get(key) is not None:
            if key == 'pca_dirname':
                if exists(join(progress_vars[key], 'pca.h5')):
                    pca_progress[key] = True
            # changepoints field only include the filename with no path and extension or abspath to the file
            elif key == 'changepoints_path':
                if dirname(abspath(progress_vars[key])) == abspath(progress_vars['pca_dirname']) and exists(progress_vars[key]):
                    pca_progress[key] = True
                # manually construct the path for changepoints.h5
                elif exists(join(progress_vars['pca_dirname'], progress_vars[key] + '.h5')):
                    pca_progress[key] = True
            else:
                if exists(progress_vars[key]):
                    pca_progress[key] = True

    return pca_progress

def get_extraction_progress(base_dir, exts=['dat', 'mkv', 'avi']):
    """
    Count the number of fully extracted sessions, and print the session directory names of the incomplete or missing extractions.

    Args:
    base_dir (str): Path to parent directory containing all sessions

    Returns:
    path_dict (dict): Dict with paths to all found sessions
    num_extracted (int): Total number of completed extractions
    """

    path_dict = get_session_paths(base_dir, exts=exts)

    e_path_dict = get_session_paths(base_dir, extracted=True, exts=exts)

    # Count number of extracted sessions and print names of the missing/incomplete extractions
    num_extracted = 0
    for k, v in path_dict.items():
        extracted_path = e_path_dict.get(k, v)
        extracted = False
        if '.mp4' in extracted_path:
            yaml_path = extracted_path.replace('mp4', 'yaml')
            if check_completion_status(yaml_path):
                extracted = True
                num_extracted += 1
            else:
                print(f'Extraction {k} is listed as incomplete.')
        if not extracted:
            print('Not yet extracted:', k)

    return path_dict, num_extracted


def print_progress(base_dir, progress_vars, exts=['dat', 'mkv', 'avi']):
    """
    Search for all the paths included in the progress file and displays 4 progress bars, one for each pipeline step.

    Args:
    base_dir (str): Path to parent directory containing all sessions
    progress_vars (dict): notebook progress dict
    """

    pca_progress = {'pca_dirname': False,
                    'scores_path': False,
                    'changepoints_path': False,
                    'index_file': False}

    modeling_progress = {'base_model_path': False}

    # Get Extract Progress
    path_dict, num_extracted = get_extraction_progress(base_dir, exts=exts)

    # Get PCA Progress
    pca_progress = get_pca_progress(progress_vars, pca_progress)

    # Get Modeling Path
    if progress_vars.get('base_model_path'):
        base_model_path = progress_vars.get('base_model_path')
        if exists(base_model_path):
            modeling_progress['model_path'] = True
            model_num = len(glob(join(base_model_path, '*.p')))

    print(f'Extraction Progress: {num_extracted} out of {len(path_dict)} session(s) extracted')

    info_print = f'PCA Progress: {sum(pca_progress.values())} out of {len(pca_progress)} items finished'
    if sum(pca_progress.values()) == len(pca_progress):
        _items = ", ".join(key for key, v in pca_progress.items() if not v)
        append_print = f': {_items} left'
        info_print += append_print
    print(info_print)

    if modeling_progress.get('model_path'):
        print(f'Found {model_num} model(s)')

def check_progress(progress_filepath=abspath('./progress.yaml'), exts=['dat', 'mkv', 'avi', 'tar.gz']):
    """
    Check whether progress file exists and prompt user input on whether to overwrite, load old, or generate a new one.

    Args:
    base_dir (str): path to directory to create/find progress file
    progress_filepath (str): path to progress filename
    """

    # Check if progress file exists
    if exists(progress_filepath):
        progress_vars = read_yaml(progress_filepath)

        print('Found progress file, displaying progress...\n')
        # Display progress bars
        print_progress(progress_vars['base_dir'], progress_vars, exts=exts)

def progress_path_sanity_check(progress_paths, progress_filepath='./progress.yaml'):
    """
    check whether all relavent paths are correct in the progress file.

    Args:
        progress_paths (dict): dictionary of the progress paths.
        progress_filepath (str, optional): path to progress.yaml file. Defaults to './progress.yaml'.
    """
    # necessary paths to check for the analysis pipeline
    must_have_paths = ['base_dir', 'config_file', 'index_file', 'train_data_dir', 'pca_dirname', 
                       'scores_filename', 'scores_path', 'changepoints_path']
    
    # keywords that should be in the paths
    keywords = [abspath(dirname(progress_filepath)), 'config.yaml', 'moseq2-index.yaml', 'aggregate_results', '_pca',
            'pca_scores','pca_scores.h5', 'changepoints']
    # zip the necessary paths and keywords for checking
    must_have_paths = dict(zip(must_have_paths, keywords))

    # sanity check for discrepancies
    for key, value in must_have_paths.items():
        if value not in progress_paths.get(key, ''):
            print(f'Please check and correct the path in {key}. The default path should contain {value}')
            print('File names are not default values, please check if this is intentional')
            print('=' * 20)

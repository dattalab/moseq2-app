'''

This module handles all jupyter notebook progress related functionalities.

'''

import os
import time
import json
import warnings
from glob import glob
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from os.path import dirname, basename
from moseq2_extract.helpers.data import check_completion_status

def get_session_paths(data_dir, extracted=False, exts=['dat', 'mkv', 'avi']):
    '''
    Find all depth recording sessions and their paths (with given extensions)
    to work on given base directory.

    Parameters
    ----------
    data_dir (str): path to directory containing all session folders.
    exts (list): list of depth file extensions to search for.

    Returns
    -------
    path_dict (dict): session directory name keys pair with their respective absolute paths.
    '''

    if extracted:
        path = '*/proc/*.'
        exts = ['mp4']
    else:
        path = '*/*.'

    sessions = []

    # Get list of sessions ending in the given extensions
    for ext in exts:
        if len(data_dir) == 0:
            data_dir = os.getcwd()
            files = sorted(glob(path + ext))
            sessions += files
        else:
            data_dir = data_dir.strip()
            if os.path.isdir(data_dir):
                files = sorted(glob(os.path.join(data_dir, path + ext)))
                sessions += files
            else:
                print('directory not found, try again.')

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': 'default', 'SessionName': 'default',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    if not extracted:
        for sess in sessions:
            # get path to session directory
            sess_dir = dirname(sess)
            sess_name = basename(sess_dir)
            # Generate metadata.json file if it's missing
            if 'metadata.json' not in os.listdir(sess_dir):
                sample_meta['SessionName'] = sess_name
                with open(os.path.join(sess_dir, 'metadata.json'), 'w') as fp:
                    json.dump(sample_meta, fp)

        # Create path dictionary
        names = [basename(dirname(sess)) for sess in sessions]
        path_dict = {n: p for n, p in zip(names, sessions)}
    else:
        names = [dirname(sess).split('/')[-2] for sess in sessions]
        path_dict = {n: p for n, p in zip(names, sessions)}

    return path_dict

def update_progress(progress_file, varK, varV):
    '''
    Updates progress file with new notebook variable

    Parameters
    ----------
    progress_file (str): path to progress file
    varK (str): key in progress file to update
    varV (str): updated value to write

    Returns
    -------
    progress (dict): Loaded path dict from the progress yaml file.
    '''

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    with open(progress_file, 'r') as f:
        progress = yaml.safe_load(f)

    progress[varK] = varV
    with open(progress_file, 'w') as f:
        yml.dump(progress, f)

    print(f'Successfully updated progress file with {varK} -> {varV}')
    return progress

def restore_progress_vars(progress_file):
    '''
    Restore all saved progress variables to Jupyter Notebook.

    Parameters
    ----------
    progress_file (str): path to progress file

    Returns
    -------
    vars (dict): All progress file variables
    '''

    warnings.filterwarnings('ignore')

    with open(progress_file, 'r') as f:
        vars = yaml.safe_load(f)

    return vars

def handle_progress_restore_input(base_progress_vars, progress_filepath):
    '''

    Helper function that handles user input for restoring progress variables.

    Parameters
    ----------
    base_progress_vars (dict): dict of default progress name to path pairs.
    progress_filepath (str): path to progress filename

    Returns
    -------
    progress_vars (dict): loaded progress variables
    '''

    restore = ''
    # Restore loaded variables or overwrite with fresh state
    while (restore != 'Y' or restore != 'N' or restore != 'q'):
        restore = input('Would you like to restore the above listed notebook variables? Y -> restore variables, N -> overwrite progress file, q -> quit]')

        if restore.lower() == "y":

            print('Updating notebook variables...')
            progress_vars = restore_progress_vars(progress_filepath)

            return progress_vars

        elif restore.lower() == "n":

            print('Overwriting progress file with initial progress.')
            progress_vars = base_progress_vars

            with open(progress_filepath, 'w') as f:
                yaml.safe_dump(progress_vars, f)

            return progress_vars

        elif restore.lower() == 'q':
            return

def print_progress(progress_vars):
    '''
    Displays tqdm progress bars checking a users jupyter notebook progress.

    Parameters
    ----------
    progress_vars (dict): notebook progress dict

    Returns
    -------
    '''

    # fill with bools for whether each session is extracted, and index file is generated

    pca_progress = {'pca_file': False, 'pca_scores': False, 'changepoints': False}
    if progress_vars.get('index_file', None) != None:
        pca_progress['index_file'] = True

    modeling_progress = {'model_path': False}
    analysis_progress = {'syll_info': False, 'crowd_dir': False}

    # Get extraction progress
    path_dict = get_session_paths(progress_vars['base_dir'])
    e_path_dict = get_session_paths(progress_vars['base_dir'], extracted=True)

    num_extracted = 0
    for k, v in path_dict.items():
        extracted_path = e_path_dict.get(k, v)
        if '.mp4' in extracted_path:
            yaml_path = extracted_path.replace('mp4', 'yaml')
            extracted = check_completion_status(yaml_path)
        else:
            extracted = False
        if extracted:
            num_extracted += 1
        else:
            print('Not yet extracted:', k)

    total_extractions = len(path_dict.keys())

    # Get PCA Progress
    if progress_vars.get('pca_dirname', None) != None:
        if os.path.exists(os.path.join(progress_vars['base_dir'], progress_vars['pca_dirname'], 'pca.h5')):
            pca_progress['pca_file'] = True
    if progress_vars.get('scores_path', None) != None:
        pca_progress['pca_scores'] = True
    if progress_vars.get('changepoints_path', None) != None:
        pca_progress['changepoints'] = True

    num_pca_files = 0
    for v in pca_progress.values():
        if v == True:
            num_pca_files += 1

    # Get Modeling Progress
    if progress_vars.get('model_path', None) != None:
        if os.path.exists(progress_vars['model_path']):
            modeling_progress['model_path'] = True

    # Get Analysis Path
    if progress_vars.get('crowd_dir', None) != None:
        if os.path.exists(progress_vars['crowd_dir']):
            analysis_progress['crowd_dir'] = True

    if progress_vars.get('syll_info', None) != None:
        if os.path.exists(progress_vars['syll_info']):
            analysis_progress['syll_info'] = True

    # Show extraction progress
    for e in tqdm(range(len(e_path_dict.keys())), total=total_extractions, desc="Extraction Progress",
                  bar_format='{desc}: {n_fmt}/{total_fmt} {bar}'):
        if e == num_extracted:
            break

    # Show PCA progress
    for j in tqdm(range(len(pca_progress.keys())), total=len(pca_progress.keys()), desc="PCA Progress",
                  bar_format='{desc}: {n_fmt}/{total_fmt} {bar}'):
        if j == num_pca_files:
            break

    # Show Modeling progress
    for i in tqdm(modeling_progress.keys(), total=len(modeling_progress.keys()), desc="Modeling Progress",
                  bar_format='{desc}: {n_fmt}/{total_fmt} {bar}'):
        if modeling_progress[i] == False:
            break

    # Show Analysis progress
    for i in tqdm(analysis_progress.keys(), total=len(analysis_progress.keys()), desc="Analysis Progress",
                  bar_format='{desc}: {n_fmt}/{total_fmt} {bar}'):
        if analysis_progress[i] == False:
            break

def check_progress(base_dir, progress_filepath):
    '''
    Checks whether progress file exists and prompts user input on whether to overwrite, load old, or generate a new one.

    Parameters
    ----------
    base_dir (str): path to directory to create/find progress file
    progress_filepath (str): path to progress filename

    Returns
    -------
    All restored variables or None.
    '''

    warnings.filterwarnings('ignore')

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    # Create basic progress file
    base_progress_vars = {'base_dir': base_dir,
                          'config_file': '',
                          'index_file': '',
                          'train_data_dir': '',
                          'pca_dirname': '',
                          'scores_filename': '',
                          'scores_path': '',
                          'model_path': '',
                          'crowd_dir': '',
                          'plot_path': os.path.join(base_dir, 'plots/')}

    # Check if progress file exists
    if os.path.exists(progress_filepath):
        with open(progress_filepath, 'r') as f:
            progress_vars = yaml.safe_load(f)

        print('Found progress file, displaying progress...\n')
        # Display progress bars
        print_progress(progress_vars)
        time.sleep(0.1)

        # Handle user input
        progress_vars = handle_progress_restore_input(base_progress_vars, progress_filepath)
        return progress_vars

    else:
        print('Progress file not found, creating new one.')
        progress_vars = base_progress_vars
        print_progress(progress_vars)

        with open(progress_filepath, 'w') as f:
            yml.dump(progress_vars, f)

        return progress_vars

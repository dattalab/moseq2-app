'''

This module handles all jupyter notebook progress related functionalities.

'''

import os
import time
import json
import warnings
from glob import glob
from time import sleep
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from os.path import dirname, basename, exists, join
from moseq2_extract.helpers.data import check_completion_status

def generate_missing_metadata(sess_dir, sess_name):
    '''
    Generates default metadata.json file for session that does not already include one.

    Parameters
    ----------
    sess_dir (str): Path to directory to create metadata.json file in.
    sess_name (str): Name of the directory to set the metadata SessionName.

    Returns
    -------

    '''

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': f'{basename(sess_dir)}', f'SessionName': f'{sess_name}',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    with open(join(sess_dir, 'metadata.json'), 'w') as fp:
        json.dump(sample_meta, fp)

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

    if len(sessions) == 0:
        if extracted:
            sessions = glob(join(data_dir, '*.mp4'))
        else:
            for ext in exts:
                sessions += glob(join(data_dir, f'*.{ext}'))

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

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

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
                yml.dump(progress_vars, f)

            return progress_vars

        elif restore.lower() == 'q':
            return

def show_progress_bar(nfound, total, desc):
    '''
    Helper function to print progress bars for each MoSeq-step progress dict

    Parameters
    ----------
    i_dict (dict): Progress dict.
    nfound (int): Total number of found progress items
    total (int): Total number of progress items
    desc (str): Progress description text to display.

    Returns
    -------
    '''

    for e in tqdm(list(range(total)), total=total, desc=desc, bar_format='{desc}: {n_fmt}/{total_fmt} {bar}'):
        sleep(0.1)
        if e == nfound:
            break

def count_total_found_items(i_dict):
    '''
    Counts the total number of found progress items

    Parameters
    ----------
    i_dict (dict): Dict containing paths to respective pipelines items.

    Returns
    -------
    num_files (int): Number of found previously computed paths.
    '''

    num_files = 0
    for v in i_dict.values():
        if v == True:
            num_files += 1

    return num_files

def get_pca_progress(progress_vars, pca_progress):
    '''
    Updates the PCA progress dict variables and prints the names of the missing keys.

    Parameters
    ----------
    progress_vars (dict): Notebook progress dict including the relevant PCA paths
    pca_progress (dict): PCA progress boolean dict used to display progress bar

    Returns
    -------
    pca_progress (dict): Updated PCA progress boolean dict.
    '''

    # Get PCA Progress
    for key in pca_progress.keys():
        if progress_vars.get(key, None) != None:
            if key == 'pca_dirname':
                if exists(join(progress_vars[key], 'pca.h5')):
                    pca_progress[key] = True
            elif key == 'changepoints_path':
                if exists(join(progress_vars['pca_dirname'], progress_vars[key] + '.h5')):
                    pca_progress[key] = True
            else:
                if exists(progress_vars[key]):
                    pca_progress[key] = True

        if pca_progress[key] != True:
            print(f'PCA missing: {key}')
    return pca_progress

def get_extraction_progress(base_dir):
    '''
    Counts the number of fully extracted sessions, and prints the session directory names
     of the incomplete or missing extractions.

    Parameters
    ----------
    base_dir (str): Path to parent directory containing all sessions

    Returns
    -------
    path_dict (dict): Dict with paths to all found sessions
    num_extracted (int): Total number of completed extractions
    '''

    path_dict = get_session_paths(base_dir)
    e_path_dict = get_session_paths(base_dir, extracted=True)

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


def print_progress(base_dir, progress_vars):
    '''
    Searches for all the paths included in the progress file and displays 4 progress bars, one for each pipeline step.

    Displays tqdm progress bars checking a users jupyter notebook progress.

    Parameters
    ----------
    base_dir (str): Path to parent directory containing all sessions
    progress_vars (dict): notebook progress dict

    Returns
    -------
    '''

    pca_progress = {'pca_dirname': False,
                    'scores_path': False,
                    'changepoints_path': False,
                    'index_file': False}

    modeling_progress = {'model_path': False}
    analysis_progress = {'syll_info': False, 'crowd_dir': False}

    # Get Extract Progress
    path_dict, num_extracted = get_extraction_progress(base_dir)

    # Get PCA Progress
    pca_progress = get_pca_progress(progress_vars, pca_progress)

    # Get Modeling Path
    if progress_vars.get('model_path', None) != None:
        if exists(progress_vars['model_path']):
            modeling_progress['model_path'] = True

    # Get Analysis Path
    if progress_vars.get('crowd_dir', None) != None:
        if exists(progress_vars['crowd_dir']):
            analysis_progress['crowd_dir'] = True

    if progress_vars.get('syll_info', None) != None:
        if exists(progress_vars['syll_info']):
            analysis_progress['syll_info'] = True

    show_progress_bar(num_extracted, len(path_dict.keys()), desc="Extraction Progress")
    show_progress_bar(count_total_found_items(pca_progress), len(pca_progress.keys()), desc="PCA Progress")
    show_progress_bar(count_total_found_items(modeling_progress), len(modeling_progress.keys()), desc="Modeling Progress")
    show_progress_bar(count_total_found_items(analysis_progress), len(analysis_progress.keys()), desc="Analysis Progress")

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
                          'changepoints_path': '',
                          'model_path': '',
                          'crowd_dir': '',
                          'syll_info': '',
                          'plot_path': os.path.join(base_dir, 'plots/')}

    # Check if progress file exists
    if exists(progress_filepath):
        with open(progress_filepath, 'r') as f:
            progress_vars = yaml.safe_load(f)

        print('Found progress file, displaying progress...\n')
        # Display progress bars
        print_progress(base_dir, progress_vars)
        time.sleep(0.1)

        # Handle user input
        progress_vars = handle_progress_restore_input(base_progress_vars, progress_filepath)
        return progress_vars

    else:
        print('Progress file not found, creating new one.')
        progress_vars = base_progress_vars
        print_progress(base_dir, progress_vars)

        with open(progress_filepath, 'w') as f:
            yml.dump(progress_vars, f)

        return progress_vars

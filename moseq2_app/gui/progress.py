'''

This module handles all jupyter notebook progress related functionalities.

'''

import os
import uuid
import json
import pickle
import logging
import warnings
from glob import glob
from time import sleep
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from datetime import datetime
from os.path import dirname, basename, exists, join
from moseq2_extract.helpers.data import check_completion_status

progress_log = 'progress.log'
progress_pkl = 'progress_log.pkl'
logging.basicConfig(filename=progress_log, level=logging.INFO)

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
            path_dict = {basename(dirname(p)): p for p in sessions}
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
            path_dict = {basename(dirname(p)): p for p in sessions}

    return path_dict

def update_pickle_log(log_dict):

    if not exists(progress_pkl):
        with open(progress_pkl, 'wb+') as fp:
            pickle.dump(log_dict, fp)
    else:
        with open(progress_pkl, 'rb') as fp:
            old_log = pickle.load(fp)

        log_dict.update(old_log)

        with open(progress_pkl, 'wb+') as fp:
            pickle.dump(log_dict, fp)

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

    if isinstance(varV, str):
        old_value = progress.get(varK, '') # get previous variable to print

        # update pickle log with latest uuid-progress key-value pair
        curr_id = str(progress.get('snapshot', uuid.uuid4()))

        log_dict = {curr_id: progress}

        if old_value != varV:
            update_pickle_log(log_dict)
        else:
            print('Variables are the same. No update necessary.')
            return progress

        progress[varK] = varV

        # update snapshot variable
        progress['snapshot'] = str(uuid.uuid4())

        with open(progress_file, 'w') as f:
            yml.dump(progress, f)

        # update log file
        logging.info(f'{datetime.now()}, {progress["snapshot"]}, {varK}: {old_value} -> {varV}')

        print(f'Successfully updated progress file with {varK} -> {varV}')
    else:
        print('Entered path is invalid.')
        print('Ensure you are updating the progress file with string paths only.')

    return progress

def generate_intital_progressfile(filename='progress.yaml'):
    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    base_dir = dirname(filename)

    print(f'Generating progress path at: {filename}')

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
                          'plot_path': os.path.join(base_dir, 'plots/'),
                          'snapshot': str(uuid.uuid4())}

    with open(filename, 'w') as f:
        yml.dump(base_progress_vars, f)

    curr_id = base_progress_vars['snapshot']
    log_dict = {curr_id: base_progress_vars}
    update_pickle_log(log_dict)

    logging.info(f'New progress file created: \n{json.dumps(base_progress_vars, indent=4)}')

    return base_progress_vars

def load_progress(progress_file):
    if exists(progress_file):
        print('Updating notebook variables...')
        with open(progress_file, 'r') as f:
            progress_vars = yaml.safe_load(f)
    else:
        print('Progress file not found.')
        progress_vars = None

    return progress_vars

def restore_progress_vars(progress_file, init=False, overwrite=False):
    '''
    Restore all saved progress variables to Jupyter Notebook.

    Parameters
    ----------
    progress_file (str): path to progress file

    Returns
    -------
    vars (dict): All progress file variables
    '''

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    # Restore loaded variables or overwrite with fresh state
    if init:
        if overwrite:
            print('Overwriting progress file with initial progress.')
            progress_vars = generate_intital_progressfile(progress_file)
        else:
            if exists(progress_file):
                progress_vars = load_progress(progress_file)
            else:
                progress_vars = generate_intital_progressfile(progress_file)
    elif overwrite:
        print('Overwriting progress file with initial progress.')
        progress_vars = generate_intital_progressfile(progress_file)
    else:
        progress_vars = load_progress(progress_file)

    return progress_vars


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
            print(f'PCA path missing: {key}')
    return pca_progress

def get_extraction_progress(base_dir, exts=['dat', 'mkv', 'avi']):
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
    analysis_progress = {'syll_info': False}

    # Get Extract Progress
    path_dict, num_extracted = get_extraction_progress(base_dir, exts=exts)

    # Get PCA Progress
    pca_progress = get_pca_progress(progress_vars, pca_progress)

    # Get Modeling Path
    if progress_vars.get('model_path', None) != None:
        if exists(progress_vars['model_path']):
            modeling_progress['model_path'] = True

    if progress_vars.get('syll_info', None) != None:
        if exists(progress_vars['syll_info']):
            analysis_progress['syll_info'] = True

    show_progress_bar(num_extracted, len(path_dict.keys()), desc="Extraction Progress")
    show_progress_bar(count_total_found_items(pca_progress), len(pca_progress.keys()), desc="PCA Progress")
    show_progress_bar(count_total_found_items(modeling_progress), len(modeling_progress.keys()), desc="Modeling Progress")
    show_progress_bar(count_total_found_items(analysis_progress), len(analysis_progress.keys()), desc="Analysis Progress")

def check_progress(base_dir, progress_filepath, exts=['dat', 'mkv', 'avi', 'tar.gz']):
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

    # Check if progress file exists
    if exists(progress_filepath):
        with open(progress_filepath, 'r') as f:
            progress_vars = yaml.safe_load(f)

        print('Found progress file, displaying progress...\n')
        # Display progress bars
        print_progress(base_dir, progress_vars, exts=exts)
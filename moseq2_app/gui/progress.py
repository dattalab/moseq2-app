"""
This module handles all jupyter notebook progress related functionalities.
"""

import os
import uuid
import json
from glob import glob, escape as glob_escape
from moseq2_common.yaml import write_yaml as _write_yaml
from moseq2_viz.util import get_sorted_index, read_yaml
from os.path import dirname, basename, exists, join, abspath
from moseq2_extract.helpers.data import check_completion_status
from collections import Counter
from pathlib import PurePath
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH, RECORDING_EXTENSIONS, find_recordings, session_keys


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


def get_sessions(data_dir, skip_extracted=True, extensions=('dat', 'mkv', 'avi', 'tar.gz'),
                 max_depth=DEFAULT_MAX_DEPTH):
    """
    Find the recordings the arena tool shows.

    The search is ``moseq2_extract.recordings.find_recordings``: one recording per session
    folder, at most max_depth folders below data_dir, following symlinked folders. Where it
    looked and what it passed over is printed when it finds nothing or leaves anything out,
    and each recording without a metadata.json is named with a warning.

    Args:
        data_dir (str): path to project directory
        skip_extracted (bool, optional): bool flag for skipping extracted sessions. Defaults to True.
        extensions (tuple, optional): extensions to look for depth recordings. Defaults to ('dat', 'mkv', 'avi', 'tar.gz').
        max_depth (int, optional): how many folders below data_dir to search. Defaults to 5.

    Returns:
        (list): sorted paths of the recordings found
    """
    search = find_recordings(data_dir, extensions=extensions, max_depth=max_depth,
                             skip_extracted=skip_extracted)
    message = search.report()
    if message:
        print(message)
    return search.recordings


def _movies(folder, flipped):
    """The movies extraction (or, with flipped, flip correction) wrote in a session folder's proc/."""
    movies = sorted(glob(join(glob_escape(folder), 'proc', '*.mp4')))
    return [m for m in movies if basename(m).endswith('_flipped.mp4') == flipped]


def _extracted_paths(data_dir, flipped=False, max_depth=DEFAULT_MAX_DEPTH):
    """
    The movie each extraction wrote, in the ``proc`` folder beside each recording.

    Recordings are found as the progress cell finds them (``find_recordings``, any recording
    type, at most max_depth folders below data_dir). A folder one level below data_dir with a
    ``proc`` folder but no recording left in it is included too, so an extraction whose
    recording was moved away is still listed. Sessions are named as ``get_session_paths``
    names them; a proc folder holding more than one movie names each by session and file.

    Args:
    data_dir (str): path to directory containing all session folders.
    flipped (bool): return the flip-corrected movies (``*_flipped.mp4``) instead.
    max_depth (int): how many folders below data_dir to search for recordings.

    Returns:
    path_dict (dict): session names paired with the absolute paths of their movies.
    """
    data_dir = abspath(data_dir.strip() or os.getcwd())
    if not os.path.isdir(data_dir):
        print(f'{data_dir} is not a folder. Check the path given as the project folder '
              '(base_dir in the notebook).')
        return {}

    search = find_recordings(data_dir, extensions=RECORDING_EXTENSIONS, max_depth=max_depth)
    folders = list(dict.fromkeys(dirname(r) for r in search.recordings))
    folders += [dirname(p) for p in sorted(glob(join(glob_escape(data_dir), '*', 'proc')))
                if dirname(p) not in folders]
    names = session_keys([join(f, 'proc') for f in folders], data_dir)

    path_dict = {}
    for folder in sorted(folders, key=lambda f: names[join(f, 'proc')]):
        name = names[join(folder, 'proc')]
        movies = _movies(folder, flipped)
        for movie in movies:
            path_dict[name if len(movies) == 1 else f'{name}/{basename(movie)}'] = movie

    if not path_dict:
        # movies kept directly in data_dir, such as a folder of copied extractions
        path_dict = {basename(p): p for p in sorted(glob(join(glob_escape(data_dir), '*.mp4')))
                     if basename(p).endswith('_flipped.mp4') == flipped}

    if not path_dict:
        movie = 'flip-corrected movie (*_flipped.mp4)' if flipped else 'extraction movie (*.mp4)'
        then = (' and apply the flip classifier to them' if flipped else '')
        print(f'No extractions found under {data_dir}. Looked for an {movie} in the proc folder beside '
              f'each of the {len(search.recordings)} recording(s) found down to {max_depth} levels below '
              f'it, and in {data_dir}. Extract the recordings first{then}, or point this cell at the '
              'project folder that holds them.')
    return path_dict


def _session_names(data_dir, recordings):
    """
    Name each recording by its session folder (``session_keys``): the folder's name, or its path
    below data_dir when two session folders share a name (``GroupA/mouse1`` and ``GroupB/mouse1``).
    Recordings sharing a folder (several archives) are each named by their own path below data_dir.
    """
    keys = session_keys(recordings, data_dir)
    counts = Counter(keys.values())
    return [keys[r] if counts[keys[r]] == 1
            else '/'.join(PurePath(os.path.relpath(r, abspath(data_dir))).parts)
            for r in recordings]


def get_session_paths(data_dir, extracted=False, flipped=False, exts=['dat', 'mkv', 'avi'],
                      max_depth=DEFAULT_MAX_DEPTH):
    """
    Find all depth recording sessions and their paths (with given extensions) to work on given base directory.

    Recordings are found with ``moseq2_extract.recordings.find_recordings``: one per session
    folder, at most max_depth folders below data_dir, following symlinked folders. Where it
    looked and what it passed over is printed when it finds nothing or leaves anything out. A
    default metadata.json is written into each session folder that has none, after a warning
    naming the folder. With extracted=True, the movie extraction wrote in the ``proc`` folder
    beside each recording is returned instead.

    Sessions are named by their folder's name, or by the folder's path below data_dir when two
    session folders share a name, as ``session_config.yaml`` names them.

    Args:
    data_dir (str): path to directory containing all session folders.
    exts (list): list of depth file extensions to search for.
    flipped (bool): indicates whether to show corrected flip videos
    extracted (bool): indicates to return paths to extracted sessions only.
    max_depth (int): how many folders below data_dir to search for recordings.

    Returns:
    path_dict (dict): session names paired with their respective absolute paths.
    """

    if extracted:
        return _extracted_paths(data_dir, flipped=flipped, max_depth=max_depth)

    data_dir = data_dir.strip() or os.getcwd()
    search = find_recordings(data_dir, extensions=exts, max_depth=max_depth)
    message = search.report()
    if message:
        print(message)
    sessions = search.recordings

    for folder, _ in search.without_metadata:
        # Generate metadata.json file if it's missing
        generate_missing_metadata(folder, basename(folder))

    return dict(zip(_session_names(data_dir, sessions), sessions))

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

    progress = read_yaml(progress_file)

    if isinstance(varV, str):
        old_value = progress.get(varK, '') # get previous variable to print        

        if old_value == varV:
            print('Variables are the same. No update necessary.')
            return progress

        progress[varK] = varV

        # update snapshot variable
        progress['snapshot'] = str(uuid.uuid4())

        _write_yaml(progress, progress_file)

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
            _write_yaml(index_params, base_progress.get('index_file'))
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
        models = sorted(models, key=os.path.getmtime, reverse=True)
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

    _write_yaml(base_progress_vars, filename)

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

    if overwrite:
        print('Overwriting progress file with initial progress.')
        # overwrite progress file with initial progress
        progress_vars = generate_intital_progressfile(progress_file)
    elif init and not exists(progress_file):
        # generate progress file if it does not exist
        progress_vars = generate_intital_progressfile(progress_file)
    else:
        # restore progress file if it exists
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

def incomplete_extraction_message(session, recording, yaml_path):
    """
    Explain an extraction whose results file says it did not finish.

    extract_wrapper writes results_XX.yaml with complete: False before extracting and sets it True
    only at the end, so a False left behind means the extraction was stopped partway.
    run_slurm_extract submits sbatch commands without an output path, so a SLURM job's log is
    sbatch's default slurm-<job id>.out in the directory the jobs were submitted from.

    Args:
    session (str): session directory name
    recording (str): path to the session's depth recording
    yaml_path (str): path to the extraction's results yaml

    Returns:
    (str): the message
    """
    proc_dir = dirname(yaml_path)
    return (
        f"Extraction of {session} started but did not finish: {yaml_path} records it as incomplete. "
        "An extraction stopped partway leaves this behind, most often because the job or notebook "
        "kernel running it was stopped, for example killed for running out of memory or for reaching "
        f"its time limit. Look in {proc_dir} for what it wrote and, if it ran on SLURM, in the job's log, "
        "slurm-<job id>.out in the directory the extraction jobs were submitted from. "
        f"Run the extraction again for {recording} to redo it."
    )


def get_extraction_progress(base_dir, exts=['dat', 'mkv', 'avi'], max_depth=DEFAULT_MAX_DEPTH):
    """
    Count the number of fully extracted sessions, and print the session directory names of the incomplete or missing extractions.

    Args:
    base_dir (str): Path to parent directory containing all sessions
    max_depth (int): how many folders below base_dir to search for recordings

    Returns:
    path_dict (dict): Dict with paths to all found sessions
    num_extracted (int): Total number of completed extractions
    """

    path_dict = get_session_paths(base_dir, exts=exts, max_depth=max_depth)

    # Count number of extracted sessions and print names of the missing/incomplete extractions
    num_extracted = 0
    for k, v in path_dict.items():
        # the movie extraction wrote beside the recording, in its proc folder
        movies = [f for f in sorted(glob(join(glob_escape(dirname(v)), 'proc', '*.mp4')))
                  if not basename(f).endswith('_flipped.mp4')]
        extracted_path = movies[-1] if movies else v
        extracted = False
        if extracted_path.endswith('.mp4'):
            yaml_path = extracted_path[:-4] + '.yaml'
            if check_completion_status(yaml_path):
                extracted = True
                num_extracted += 1
            else:
                print(incomplete_extraction_message(k, v, yaml_path))
        if not extracted:
            print('Not yet extracted:', k)

    return path_dict, num_extracted


def print_progress(base_dir, progress_vars, exts=['dat', 'mkv', 'avi'], max_depth=DEFAULT_MAX_DEPTH):
    """
    Search for all the paths included in the progress file and displays 4 progress bars, one for each pipeline step.

    Args:
    base_dir (str): Path to parent directory containing all sessions
    progress_vars (dict): notebook progress dict
    max_depth (int): how many folders below base_dir to search for recordings
    """

    pca_progress = {'pca_dirname': False,
                    'scores_path': False,
                    'changepoints_path': False,
                    'index_file': False}

    modeling_progress = {'base_model_path': False}

    # Get Extract Progress
    path_dict, num_extracted = get_extraction_progress(base_dir, exts=exts, max_depth=max_depth)

    # Get PCA Progress
    pca_progress = get_pca_progress(progress_vars, pca_progress)

    # Get Modeling Path
    if progress_vars.get('base_model_path'):
        base_model_path = progress_vars.get('base_model_path')
        if exists(base_model_path):
            modeling_progress['model_path'] = True
            model_num = len(glob(join(base_model_path, '*.p')))

    # with no recordings, the search has already said where it looked and why it found none
    if path_dict:
        print(f'Extraction Progress: {num_extracted} out of {len(path_dict)} session(s) extracted')

    info_print = f'PCA Progress: {sum(pca_progress.values())} out of {len(pca_progress)} items finished'
    if sum(pca_progress.values()) != len(pca_progress):
        _items = ", ".join(key for key, v in pca_progress.items() if not v)
        append_print = f': {_items} left'
        info_print += append_print
    print(info_print)

    if modeling_progress.get('model_path'):
        print(f'Found {model_num} model(s)')

def check_progress(progress_filepath=abspath('./progress.yaml'), exts=['dat', 'mkv', 'avi', 'tar.gz'],
                   max_depth=DEFAULT_MAX_DEPTH):
    """
    Check whether progress file exists and prompt user input on whether to overwrite, load old, or generate a new one.

    Args:
    progress_filepath (str): path to progress filename
    exts (list): recording extensions to search for
    max_depth (int): how many folders below the project folder to search for recordings
    """

    # Check if progress file exists
    if exists(progress_filepath):
        progress_vars = read_yaml(progress_filepath)

        print('Found progress file, displaying progress...\n')
        # Display progress bars
        print_progress(progress_vars['base_dir'], progress_vars, exts=exts, max_depth=max_depth)

def progress_path_sanity_check(progress_paths, progress_filepath='./progress.yaml'):
    """
    Check that every file the analysis notebook reads is where the progress file says.

    Covers the model, the folder of models, the index and the config named in the progress
    file, and the PCA scores and each session's extraction results named in the index. Prints
    one line per missing path and nothing when all of them exist.

    Args:
        progress_paths (dict): dictionary of the progress paths.
        progress_filepath (str, optional): path to progress.yaml file. Not read; accepted so that
            notebooks which pass it still run.
    """
    missing = []
    for key in ['base_model_path', 'model_path', 'index_file', 'config_file']:
        path = progress_paths.get(key)
        if not path or not exists(path):
            missing.append((key, path))

    index_file = progress_paths.get('index_file')
    if index_file and exists(index_file):
        try:
            sorted_index = get_sorted_index(index_file)
        except Exception as error:
            print(f'Please check the index file {index_file}, which could not be read: {error}')
        else:
            if not exists(sorted_index['pca_path']):
                missing.append((f'pca_path in {index_file}', sorted_index['pca_path']))
            for uuid, entry in sorted_index['files'].items():
                session = entry.get('metadata', {}).get('SessionName', uuid)
                if not exists(entry['path'][0]):
                    missing.append((f'session {session} in {index_file}', entry['path'][0]))

    for where, path in missing:
        print(f'Please check and correct the path in {where}. {path} does not exist.')

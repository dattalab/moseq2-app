"""
Finding a project's recordings and extractions, for the notebooks' project, arena and QC cells.
"""

import os
import json
from glob import glob, escape as glob_escape
from os.path import dirname, basename, join, abspath
from moseq2_extract.helpers.data import check_completion_status
from collections import Counter
from pathlib import PurePath
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH, RECORDING_EXTENSIONS, find_recordings
from moseq2_common.project import outdated_names

#: The 1.5 progress functions the 1.5 notebooks import: each stops with a message to download the 1.6
#: notebooks, which start from ``moseq2_app.project.start``.
__getattr__ = outdated_names(__name__, ("restore_progress_vars", "check_progress", "update_progress",
                                        "progress_path_sanity_check"))


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
    names = _folder_names([join(f, 'proc') for f in folders], data_dir)

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


def _folder_names(paths, root):
    """
    Name the folder each of ``paths`` is in, as the notebook's lists show it.

    A folder is named by its own name when no other path's folder has that name;
    folders that share a name (``GroupA/mouse1`` and ``GroupB/mouse1``) are each
    named by their path below ``root``, with ``/`` between folders on every system.

    Args:
    paths (iterable): paths of files.
    root (str): the project folder they were found under.

    Returns:
    (dict): each path's folder name, by path.
    """
    root = abspath(root)
    folders = {p: abspath(dirname(abspath(p))) for p in paths}
    owners = {}
    for folder in set(folders.values()):
        owners.setdefault(basename(folder), []).append(folder)

    def name(folder):
        if len(owners[basename(folder)]) == 1 or folder == root:
            return basename(folder)
        return '/'.join(PurePath(os.path.relpath(folder, root)).parts)

    return {p: name(f) for p, f in folders.items()}


def _session_names(data_dir, recordings):
    """
    Name each recording by its session folder (``_folder_names``): the folder's name, or its path
    below data_dir when two session folders share a name (``GroupA/mouse1`` and ``GroupB/mouse1``).
    Recordings sharing a folder (several archives) are each named by their own path below data_dir.
    """
    keys = _folder_names(recordings, data_dir)
    counts = Counter(keys.values())
    return [keys[r] if counts[keys[r]] == 1
            else '/'.join(PurePath(os.path.relpath(r, abspath(data_dir))).parts)
            for r in recordings]


def get_session_paths(data_dir, extracted=False, flipped=False, exts=['dat', 'mkv', 'avi'],
                      max_depth=DEFAULT_MAX_DEPTH, write_missing_metadata=True):
    """
    Find all depth recording sessions and their paths (with given extensions) to work on given base directory.

    Recordings are found with ``moseq2_extract.recordings.find_recordings``: one per session
    folder, at most max_depth folders below data_dir, following symlinked folders. Where it
    looked and what it passed over is printed when it finds nothing or leaves anything out. A
    default metadata.json is written into each session folder that has none, after a warning
    naming the folder. With extracted=True, the movie extraction wrote in the ``proc`` folder
    beside each recording is returned instead.

    Sessions are named by their folder's name, or by the folder's path below data_dir when two
    session folders share a name.

    Args:
    data_dir (str): path to directory containing all session folders.
    exts (list): list of depth file extensions to search for.
    flipped (bool): indicates whether to show corrected flip videos
    extracted (bool): indicates to return paths to extracted sessions only.
    max_depth (int): how many folders below data_dir to search for recordings.
    write_missing_metadata (bool): write a default metadata.json into each session folder that has none.

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

    for folder, _ in search.without_metadata if write_missing_metadata else ():
        # Generate metadata.json file if it's missing
        generate_missing_metadata(folder, basename(folder))

    return dict(zip(_session_names(data_dir, sessions), sessions))

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


def get_extraction_progress(base_dir, exts=['dat', 'mkv', 'avi'], max_depth=DEFAULT_MAX_DEPTH,
                            write_missing_metadata=True):
    """
    Count the number of fully extracted sessions, and print the session directory names of the incomplete or missing extractions.

    Args:
    base_dir (str): Path to parent directory containing all sessions
    max_depth (int): how many folders below base_dir to search for recordings

    Returns:
    path_dict (dict): Dict with paths to all found sessions
    num_extracted (int): Total number of completed extractions
    """

    path_dict = get_session_paths(base_dir, exts=exts, max_depth=max_depth,
                                  write_missing_metadata=write_missing_metadata)

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

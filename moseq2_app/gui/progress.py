"""
Finding a project's recordings and extractions, for the notebooks' project, arena and QC cells.
"""

import os
from glob import glob, escape as glob_escape
from os.path import dirname, basename, join, abspath
from collections import Counter
from pathlib import PurePath
from moseq2_common.bookkeeping import load
from moseq2_common.project import Project, outdated_names, session_key
from moseq2_extract.recordings import (DEFAULT_MAX_DEPTH, RECORDING_EXTENSIONS, extracted_in, find_recordings,
                                       session_outputs)

#: The 1.5 progress functions the 1.5 notebooks import: each stops with a message to download the 1.6
#: notebooks, which start from ``moseq2_app.project.start``.
__getattr__ = outdated_names(__name__, ("restore_progress_vars", "check_progress", "update_progress",
                                        "progress_path_sanity_check"))


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
                             extracted=extracted_in(Project.at(data_dir)) if skip_extracted else None)
    message = search.report()
    if message:
        print(message)
    return search.recordings


def _movie(outputs, flipped):
    """A session's extraction movie, or with flipped its flip-corrected movie (``results_flipped.mp4``)."""
    return outputs.movie.with_name(f"{outputs.movie.stem}_flipped.mp4") if flipped else outputs.movie


def _session_outputs(data_dir, max_depth=DEFAULT_MAX_DEPTH):
    """Each session of the project in ``data_dir`` with its outputs (``session_outputs``), by the name the
    notebook's lists give it: its recording's folder (``_session_names``), or the session key of a session
    moseq2-project.yaml lists whose recording is no longer in the project."""
    project = Project.at(data_dir)
    recordings = find_recordings(data_dir, extensions=RECORDING_EXTENSIONS, max_depth=max_depth).recordings
    keys = [session_key(project, r) for r in recordings]
    named = dict(zip(keys, _session_names(data_dir, recordings)))
    for key in load(project).sessions:
        named.setdefault(key, key)
    return {name: session_outputs(project, key, keys=keys) for key, name in named.items()}


def _extracted_paths(data_dir, flipped=False, max_depth=DEFAULT_MAX_DEPTH):
    """
    The movie each session's extraction wrote, in its folder in the project's ``aggregate_results``.

    Sessions are the project's recordings (found as the progress cell finds them) and the sessions
    its moseq2-project.yaml lists, named as ``get_session_paths`` names them. A folder that is not a
    project, such as ``aggregate_results`` itself, lists the movies in it and in its folders, each
    named by its path below the folder.

    Args:
    data_dir (str): the project folder, or a folder holding extractions.
    flipped (bool): return the flip-corrected movies (``results_flipped.mp4``) instead.
    max_depth (int): how many folders below data_dir to search for recordings.

    Returns:
    path_dict (dict): session names paired with the absolute paths of their movies.
    """
    data_dir = abspath(data_dir.strip() or os.getcwd())
    if not os.path.isdir(data_dir):
        print(f'{data_dir} is not a folder. Check the path given as the project folder '
              '(base_dir in the notebook).')
        return {}

    path_dict = {name: str(_movie(outputs, flipped))
                 for name, outputs in sorted(_session_outputs(data_dir, max_depth).items())
                 if _movie(outputs, flipped).is_file()}

    if not path_dict:
        # a folder of extractions, such as aggregate_results
        movies = sorted(glob(join(glob_escape(data_dir), '*.mp4')) + glob(join(glob_escape(data_dir), '*', '*.mp4')))
        path_dict = {'/'.join(PurePath(os.path.relpath(p, data_dir)).parts): p for p in movies
                     if basename(p).endswith('_flipped.mp4') == flipped}

    if not path_dict:
        movie = 'flip-corrected movie (results_flipped.mp4)' if flipped else 'extraction movie (results.mp4)'
        then = (' and apply the flip classifier to them' if flipped else '')
        print(f'No extractions found under {data_dir}. Looked for an {movie} in the folder of each session in '
              f'{join(data_dir, "aggregate_results")}, and in {data_dir} and its folders. Extract the '
              f'recordings first{then}, or point this cell at the project folder that holds them.')
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
                      max_depth=DEFAULT_MAX_DEPTH):
    """
    Find all depth recording sessions and their paths (with given extensions) to work on given base directory.

    Recordings are found with ``moseq2_extract.recordings.find_recordings``: one per session
    folder, at most max_depth folders below data_dir, following symlinked folders. Where it
    looked and what it passed over is printed when it finds nothing or leaves anything out, with a
    warning naming each recording that has no metadata.json; nothing is written. With
    extracted=True, the movie extraction wrote in each session's folder is returned instead.

    Sessions are named by their folder's name, or by the folder's path below data_dir when two
    session folders share a name.

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

    return dict(zip(_session_names(data_dir, sessions), sessions))

def incomplete_extraction_message(session, recording, partial):
    """
    Explain an extraction that started but did not finish.

    extract_wrapper writes the session's ``.h5`` under a temporary name and renames it to
    ``results.h5`` only at the end, so the temporary file left behind means the extraction was
    stopped partway. run_slurm_extract submits sbatch commands without an output path, so a SLURM
    job's log is sbatch's default slurm-<job id>.out in the directory the jobs were submitted from.

    Args:
    session (str): the session's name
    recording (str): path to the session's depth recording
    partial (str): path to the unfinished extraction file

    Returns:
    (str): the message
    """
    folder = dirname(partial)
    return (
        f"Extraction of {session} started but did not finish: {partial} is the unfinished extraction. "
        "An extraction stopped partway leaves this behind, most often because the job or notebook "
        "kernel running it was stopped, for example killed for running out of memory or for reaching "
        f"its time limit. Look in {folder} for what it wrote and, if it ran on SLURM, in the job's log, "
        "slurm-<job id>.out in the directory the extraction jobs were submitted from. "
        f"Run the extraction again for {recording} to redo it."
    )


def get_extraction_progress(base_dir, exts=['dat', 'mkv', 'avi'], max_depth=DEFAULT_MAX_DEPTH):
    """
    Count the number of fully extracted sessions, and print the session directory names of the incomplete or missing extractions.

    A session is extracted when its ``results.h5`` exists in its folder in the project's ``aggregate_results``.

    Args:
    base_dir (str): Path to parent directory containing all sessions
    max_depth (int): how many folders below base_dir to search for recordings

    Returns:
    path_dict (dict): Dict with paths to all found sessions
    num_extracted (int): Total number of completed extractions
    """

    path_dict = get_session_paths(base_dir, exts=exts, max_depth=max_depth)
    project = Project.at(abspath(base_dir.strip() or os.getcwd()))
    keys = [session_key(project, v) for v in path_dict.values()]

    # Count number of extracted sessions and print names of the missing/incomplete extractions
    num_extracted = 0
    for (k, v), key in zip(path_dict.items(), keys):
        outputs = session_outputs(project, key, keys=keys)
        partial = outputs.folder / f".{outputs.h5.name}.partial"
        if outputs.h5.is_file():
            num_extracted += 1
            continue
        if partial.is_file():
            print(incomplete_extraction_message(k, v, str(partial)))
        print('Not yet extracted:', k)

    return path_dict, num_extracted

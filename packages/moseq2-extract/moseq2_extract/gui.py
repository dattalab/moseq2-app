"""
GUI front-end operations accessible from a jupyter notebook.

Each function reads the ``extract`` section of the project's ``config.yaml``,
resolves the step's settings and calls the step in ``moseq2_extract.steps`` (or
``moseq2_extract.setup``).
"""

import os
import re
from dataclasses import replace
from os.path import exists, join

from moseq2_common.options import read_settings
from moseq2_common.project import find_project
from moseq2_extract import setup, steps
from moseq2_extract.options import EXTRACT_STAGE, AggregateOptions, BatchExtractOptions
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH, RECORDING_EXTENSIONS
from moseq2_extract.session_config import recordings_options
from moseq2_extract.util import filter_warnings


_SELECTION_ITEM = re.compile(r"(e?)\s*(\d+)(?:\s*-\s*(\d+))?", re.IGNORECASE)


def _parse_session_selection(answer, n_sessions):
    """
    Turn an answer at the session prompt into the session numbers it selects.

    The answer is a comma separated list of items. Each item is a session number
    "N", an inclusive range "N-M", or either of those prefixed with "e" to leave
    those sessions out. Without an item that selects sessions, the answer starts
    from all of them. Sessions are numbered from 1.

    Args:
    answer (str): what the user typed.
    n_sessions (int): number of sessions listed.

    Returns:
    selected (list): session numbers in the order first named, each once.

    Raises:
    ValueError: naming the item that is not a number or a range, names a session
    that is not listed or runs backwards, or the answer if it leaves no sessions.
    """
    items = [item.strip() for item in answer.split(",") if item.strip()]
    if not items:
        raise ValueError("Invalid input.")

    selected, excluded = [], set()
    for item in items:
        match = _SELECTION_ITEM.fullmatch(item)
        if match is None:
            raise ValueError(f'Invalid input "{item}": not a number or a range.')
        first, last = int(match[2]), int(match[3] or match[2])
        for number in (first, last):
            if not 1 <= number <= n_sessions:
                raise ValueError(f'Invalid input "{item}": there is no session {number}.')
        if first > last:
            raise ValueError(f'Invalid input "{item}": a range runs from the lower number to the higher.')
        if match[1]:
            excluded.update(range(first, last + 1))
        else:
            selected += [n for n in range(first, last + 1) if n not in selected]

    selected = [n for n in selected or range(1, n_sessions + 1) if n not in excluded]
    if not selected:
        raise ValueError(f'Invalid input "{answer.strip()}": that leaves no sessions to extract.')
    return selected


def get_selected_sessions(to_extract, extract_all):
    """
    Return either selected sessions to extract, or all the sessions given user input, the function will

    Args:
    to_extract (list): list of paths to sessions to extract
    extract_all (bool): boolean to include all sessions and skip user-input prompt.

    Returns:
    to_extract (list): new list of selected sessions to extract.
    """

    if len(to_extract) > 1 and not extract_all:
        for i, sess in enumerate(to_extract):
            print(f"[{str(i + 1)}] {sess}")

        print("You may input comma separated values for individual sessions")
        print(
            'Or you can input a hyphen separated range. E.g. "1-10" selects 10 sessions, including sessions 1 and 10'
        )
        print(
            'You can also exclude a range by prefixing the range selection with the letter "e"; e.g.: "e1-5".'
        )
        print("Press q to quit.")
        while True:
            sessions = input("Input your selected sessions to extract: ")
            if sessions.strip().lower() == "q":
                return []
            try:
                return [to_extract[n - 1] for n in _parse_session_selection(sessions, len(to_extract))]
            except ValueError as error:
                print(f"{error} Try again or press q to quit.")
    else:
        return to_extract


@filter_warnings
def generate_config_command(output_file, camera_type="auto"):
    """
    Generate configuration file (config.yaml) to use throughout pipeline.

    Args:
    output_file (str): path to saved config file.
    camera_type (str): ``auto``, ``kinect`` (``k2``) or ``azure``, as ``moseq2-extract generate-config
        --camera-type`` takes it.

    Returns:
    (str): status message.
    """

    # Check if the file already exists, and prompt user if they would like to overwrite pre-existing file
    if exists(output_file):
        ow = input(
            "This file already exists, would you like to overwrite it? [y -> yes, n -> no] "
        )
        if ow.lower() != "y":
            return "Configuration file has been retained"
    else:
        print("Creating configuration file.")
    setup.generate_config(output_file, camera_type, overwrite=True)

    return "Configuration file has been successfully generated."


@filter_warnings
def extract_found_sessions(
    input_dir,
    config_file,
    ext=RECORDING_EXTENSIONS,
    extract_all=True,
    skip_extracted=False,
    max_depth=DEFAULT_MAX_DEPTH,
    run_cmd=False,
):
    """
    Find and extract the depth recordings within input_dir

    Args:
    input_dir (str): path to directory containing all session folders
    config_file (str): path to config file
    ext (str or list): file extension(s) for depth files to search for. Defaults to every type the
        pipeline reads (.dat, .mkv, .avi, .tar.gz); None means the same.
    extract_all (bool): if True, auto searches for all sessions, else, prompts user to select sessions individually.
    skip_extracted (bool): leave out the recordings already extracted, as ``moseq2-extract batch-extract``
        does without ``--skip-checks``.
    max_depth (int): how many folders below input_dir to search for recordings.
    run_cmd (bool): with cluster_type slurm, submit the extraction jobs; otherwise they are only
        written to the script for you to run.
    """
    # error out early
    if not exists(config_file):
        raise IOError(f"Config file {config_file} does not exist")

    config = read_settings(config_file, BatchExtractOptions, stage_classes=EXTRACT_STAGE)
    opts = BatchExtractOptions.resolve(config=config, explicit={
        "extensions": RECORDING_EXTENSIONS if ext is None else ext,
        "max_depth": max_depth,
        "skip_checks": not skip_extracted,
        "skip_completed": skip_extracted,
        "run_cmd": run_cmd,
    })
    result = steps.batch_extract(
        input_dir, opts, config_file=config_file,
        select=lambda found: get_selected_sessions(found, extract_all),
        project=find_project(config_file),
    )
    # the local extraction ends with a summary naming each recording that failed
    if opts.cluster_type == "local" and result.found and not result.failures:
        print("Extractions Complete.")


@filter_warnings
def aggregate_extract_results_command(
    project_folder, format="{start_time}_{session_name}_{subject_name}", output_dir="aggregate_results",
    mouse_threshold=None,
):
    """
    Copy every finished extraction in the project into ``output_dir``, named from its metadata, and list
    each as one of the project's sessions in its bookkeeping file (moseq2-project.yaml).

    Args:
    project_folder (str): the project folder, holding config.yaml and the recordings.
    format (str): filename format for info to include in filenames
    output_dir (str): the folder to copy the extractions into, relative to the project folder.
    mouse_threshold (float): min threshold of extracted mouse height to include a session
    in the aggregated sesssions to ensure sessions with no extracted mouse is not included in the aggregated sessions.
    None takes config.yaml's ``extract`` section, else ``AggregateOptions``' default.

    Returns:
    sessions (list): the sessions this run added, by key.
    """

    # the project's folder, where the command line's is the working directory's (PR 4.5)
    output_dir = join(project_folder, output_dir)

    if not exists(output_dir):
        os.makedirs(output_dir)

    project = find_project(cwd=project_folder)
    config = read_settings(project.config_file if project is not None else None,
                           AggregateOptions, stage_classes=EXTRACT_STAGE)
    explicit = {"format": format}
    if mouse_threshold is not None:
        explicit["mouse_threshold"] = mouse_threshold
    opts = AggregateOptions.resolve(config=config, explicit=explicit)
    return steps.aggregate_results(project_folder, output_dir, opts, project=project).sessions


def download_flip_command(output_dir, config_file="", selection=1):
    """
    Download flip classifier and saves its path to config file (config.yaml)

    Args:
    output_dir (str): path to output directory to save flip classifier
    config_file (str): path to config file (config.yaml)
    selection (int): index of which flip file to download (default is Adult male C57 classifer)

    Returns:
    """

    setup.download_flip_file(config_file, output_dir, selection)


@filter_warnings
def extract_command(
    input_file,
    output_dir,
    config_file,
    num_frames=None,
    skip=False,
    cli_options=None,
):
    """
    Extract depth file

    The recording is extracted with the settings ``recordings_options`` resolves: the arena
    tool's settings from its entry in the project's session_config.yaml, every other setting
    from the config file, and the typed options over both.

    Args:
    input_file (str): path to depth file to extract.
    output_dir (str): path to output directory.
    config_file (str): path to config file (config.yaml).
    num_frames (int): number of frames to extract. If None, all frames are extracted.
    skip (bool): skip already extracted file.
    cli_options (dict): parameters typed on the command line; they take precedence over the config file and the session config.

    Returns:
    (str or None): path to the output directory, or None if the file was skipped as already extracted.
    """

    resolved = recordings_options(config_file, [input_file], cli_options)
    opts = replace(resolved[input_file], num_frames=num_frames, skip_completed=skip)
    result = steps.extract(input_file, output_dir, opts)

    return None if result.h5_file is None else str(result.h5_file.parent)

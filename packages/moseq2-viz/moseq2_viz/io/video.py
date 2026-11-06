"""
Helper functions for handling crowd movie file writing and video metadata maintenance.
"""

import warnings
import h5py
import numpy as np
import os
from os.path import join
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import moseq2_common.video
from moseq2_common.timestamps import frame_rate
from moseq2_viz.viz import make_crowd_matrix
from cytoolz.itertoolz import first
from cytoolz.dicttoolz import valfilter
from moseq2_viz.model.util import get_syllable_slices
from cytoolz.curried import valmap
from moseq2_common.yaml import write_yaml


def check_video_parameters(index: dict) -> dict:
    """
    The frame size and frame rate the sessions' extractions share, read from each session's ``.h5``: the
    size of its frames, and the recording's frame rate the extraction stores
    (``moseq2_common.timestamps.frame_rate``). Raises a RuntimeError when the sessions differ.

    Args:
    index (dict): the sessions, as the sorted index holds them.

    Returns:
    vid_parameters (dict): ``crop_size``, ``fps``, and ``resolution`` (None).
    """
    found = {"crop_size": set(), "fps": set()}
    for entry in index["files"].values():
        with h5py.File(entry["path"][0], "r") as f:
            found["crop_size"].add(tuple(f["frames"].shape[1:]))
            found["fps"].add(frame_rate(f))

    incorrect_parameters = valfilter(lambda x: len(x) > 1, found)

    # if there are multiple values for a parameter, raise error
    if incorrect_parameters:
        raise RuntimeError(
            "The following parameters are not equal "
            + f"across extractions: {list(incorrect_parameters)}"
        )

    vid_parameters = valmap(first, found)
    vid_parameters["resolution"] = None
    return vid_parameters


def write_crowd_movie_info_file(model_path, model_fit, index_file, output_dir):
    """
    Create an info.yaml file in the crowd movie directory that holds model training parameters.

    Args:
    model_path (str): path to model used to generate movies
    model_fit (dict): loaded ARHMM mdoel results
    index_file (str or dict): path to the index file used with the model, or the sessions in memory
        (``moseq2_common.bookkeeping.session_index``)
    output_dir (str): path to crowd movies directory to store file in.
    """

    # Crowd movie info file contents; used to indicate the modeling state the crowd_movies were generated from
    info_parameters = ["model_class", "kappa", "gamma", "alpha"]

    # Loading parameters to dict to save to file in output directory
    info_file = join(output_dir, "info.yaml")
    info_dict = {k: model_fit["model_parameters"][k] for k in info_parameters}

    # Adding model and index file paths
    info_dict["model_path"] = model_path
    # sessions held in memory name the bookkeeping file they were read from
    info_dict["index_path"] = (str(index_file) if isinstance(index_file, (str, os.PathLike))
                               else index_file.get("bookkeeping_file"))

    # Convert numpy dtypes to their corresponding primitives
    info_dict = valmap(
        lambda v: v.item() if isinstance(v, np.generic) else v, info_dict
    )

    # Write metadata info file
    write_yaml(info_dict, info_file)


def write_crowd_movies(
    sorted_index, config_data, ordering, labels, label_uuids, output_dir
):
    """
    Create crowd movies for each syllable.

    Args:
    sorted_index (dict): dictionary of sorted index data.
    config_data (dict): dictionary of visualization parameters.
    filename_format (str): string format that denotes the saved crowd movie file names.
    ordering (list): ordering for the new mapping of the relabeled syllable usages.
    labels (numpy ndarray): list of syllable usages
    label_uuids (list): list of session uuids each series of labels belongs to.
    output_dir (str): path directory where all the movies are written.
    """
    progress_bar = config_data.get("progress_bar", False)

    # Filtering parameters
    clean_params = {
        "gaussfilter_space": config_data["gaussfilter_space"],
        "medfilter_space": config_data["medfilter_space"],
    }

    # Set crowd movie filename format based on whether syllables were relabeled
    if config_data.get("sort", True):
        filename_format = "syllable_sorted-id-{:02d}_({})_original-id-{:02d}.mp4"
    else:
        filename_format = "syllable_{:02d}.mp4"

    # Ensure all video metadata parameters are consistent
    vid_parameters = check_video_parameters(sorted_index)
    if vid_parameters["resolution"] is not None:
        config_data["raw_size"] = vid_parameters["resolution"]
    # writing function
    config_data["fps"] = vid_parameters["fps"]

    # Get frame slices from all included sessions for each syllable label
    slice_fun = partial(
        get_syllable_slices, labels=labels, label_uuids=label_uuids, index=sorted_index
    )

    # create crowd movie matrix to put the examples in the same syllable into a movie
    matrix_fun = partial(
        make_crowd_matrix,
        nexamples=config_data.get("max_examples", 20),
        max_dur=config_data.get("max_dur", 60),
        min_dur=config_data.get("min_dur", 0),
        min_height=config_data.get("min_height", 10),
        crop_size=vid_parameters.get("crop_size", (80, 80)),
        raw_size=config_data.get("raw_size", (512, 424)),
        select_median_duration_instances=config_data.get(
            "select_median_duration_instances", False
        ),
        scale=config_data.get("scale", 1),
        pad=config_data.get("pad", 30),
        frame_path=config_data.get("frame_path", "frames"),
        legacy_jitter_fix=config_data.get("legacy_jitter_fix", False),
        seed=config_data.get("seed", 0),
        **clean_params,
    )

    # write the crowd movies
    write_fun = partial(
        moseq2_common.video.write_frames_preview,
        fps=vid_parameters["fps"],
        depth_min=config_data["min_height"],
        depth_max=config_data["max_height"],
        cmap=config_data["cmap"],
        progress_bar=progress_bar,
    )

    namer = partial(
        _fname_formatter,
        format=filename_format,
        output_dir=output_dir,
        ordering=ordering,
        count=config_data["count"],
    )

    make_matrix = partial(
        _matrix_writer_helper,
        matrix_fun=matrix_fun,
        slice_fun=slice_fun,
        write_fun=write_fun,
        namer=namer,
    )

    # parallel process the crowd movies for all syllables. The workers start
    # fresh ("spawn", as they always do on macOS and Windows) rather than as
    # forked copies of the caller: a fork copies the threads of JAX and
    # OpenBLAS mid-flight, and with the OpenBLAS SciPy 1.17 bundles the
    # caller's next multithreaded LAPACK call then deadlocks.
    with mp.get_context("spawn").Pool(config_data.get("processes")) as pool:
        # Compute crowd matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return list(
                tqdm(
                    filter(
                        lambda x: x is not None,
                        pool.imap(make_matrix, config_data["crowd_syllables"]),
                    ),
                    desc="Writing crowd movies",
                    total=len(config_data["crowd_syllables"]),
                )
            )


def _fname_formatter(syll, format, output_dir, ordering, count):
    """
    Create filename strings for the syllable crowd movies to generate.

    Args:
    syll (int): syllable number.
    format (str): format string for outputted file. E.g. "syllable_sorted-id-{:02d}_({})_original-id-{:02d}.mp4"
    output_dir (str): path to output directory containing the crowd movie.
    ordering (dict): dictionary that holds the original id of the syllable crowd movie (pre-reordering).
    count (str): method to compute syllable mean usage, either 'usage' or 'frames'.

    Returns:
    (str): path to syllable crowd movie.
    """
    return join(output_dir, format.format(syll, count, ordering[syll]))


def _matrix_writer_helper(syll, matrix_fun, slice_fun, write_fun, namer):
    """
    Generate crowd movies using multiprocessing.

    Args:
    syll (int): syllable number.
    matrix_fun (function): helper function to create stacked video matrices given syllable slices.
    slice_fun (function): helper function to generate syllable slices given syllable number.
    write_fun (function): helper function to write the crowd movies to their respective files.
    namer (function): helper function to create filename strings for the syllable crowd movies to generate.

    Returns:
    (function): helper function to write the crowd movies to their respective files, if the video matrices were created.
    """

    mtx = matrix_fun(slice_fun(syll))
    if mtx is not None:
        return write_fun(namer(syll), mtx)
    return None

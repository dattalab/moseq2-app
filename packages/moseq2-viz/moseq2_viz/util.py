"""
General utility functions fors loading and organizing data.
"""

import re
import os
import h5py
import numpy as np
from glob import glob
from typing import Union
from cytoolz import curry, compose
from cytoolz.curried import valmap
from cytoolz.dicttoolz import dissoc, assoc
from cytoolz.itertoolz import first, groupby
from os.path import join, exists, dirname, splitext
import moseq2_common.h5
import moseq2_common.yaml


def get_index_hits(config_data, metadata, key, v):
    """
    Search for matching keys in given index file metadata dict and return a list of booleans indicating that a session was found.

    Args:
    config_data (dict): dictionary containing boolean search filters [lowercase, negative]
    metadata (list): list of session metadata dict objects
    key (str): metadata key being searched for
    v (str): value of the corresponding key to be found

    Returns:
    hits (list): list of booleans indicating the found sessions to be updated in add_group_wrapper()
    """

    if config_data["lowercase"] and config_data["negative"]:
        # Convert keys to lowercase and return inverse selection
        hits = [re.search(v, meta[key].lower()) is None for meta in metadata]
    elif config_data["lowercase"]:
        # Convert keys to lowercase
        hits = [re.search(v, meta[key].lower()) is not None for meta in metadata]
    elif config_data["negative"]:
        # Return inverse selection
        hits = [re.search(v, meta[key]) is None for meta in metadata]
    else:
        # Default search
        hits = [re.search(v, meta[key]) is not None for meta in metadata]

    return hits


def clean_dict(dct):
    """
    Cast numpy.array into lists and np.generic data into scalar values.

    Args:
    dct (dict): dictionary with values to clean.

    Returns:
    (dict): dictionary with standardized value types.
    """

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            # np.asscalar was removed in numpy 1.23; .item() is what it called.
            out = e.item()
        else:
            out = e
        return out

    return valmap(clean_entry, dct)


def get_metadata_path(h5file):
    """
    Return path within h5 file that contains the kinect extraction metadata.

    Args:
    h5file (str): path to h5 file.

    Returns:
    (str): path to acquistion metadata within h5 file.
    """

    with h5py.File(h5file, "r") as f:
        if "/metadata/acquisition" in f:
            return "/metadata/acquisition"
        elif "/metadata/extraction" in f:
            return "/metadata/extraction"
        else:
            raise KeyError("acquisition metadata not found")


def load_changepoint_distribution(cpfile):
    """
    Load changepoint durations from given model free changepoints file.

    Args:
    cpfile (str): Path to changepoints h5 file.

    Returns:
    (numpy.array): Array of changepoint durations.
    """

    cps = moseq2_common.h5.h5_to_dict(cpfile, "cps")
    cp_dist = map(compose(np.diff, np.squeeze), cps.values())
    return np.concatenate(list(cp_dist))


def changepoint_frame_rates(cpfile, uuids, model=None) -> list:
    """
    The frame rate of each session, as the changepoints file records it: the
    ``fps`` attribute of the session's ``cps``, or the file's ``metadata/fps``.

    Args:
    cpfile (str): Path to changepoints h5 file.
    uuids (list): the sessions' uuids.
    model (str): the model whose sessions these are, for the error message.

    Returns:
    rates (list of float): frames per second, one per uuid.
    """
    rates = []
    with h5py.File(cpfile, "r") as f:
        shared = f["metadata/fps"][()] if "metadata/fps" in f else None
        for uuid in uuids:
            path = f"cps/{uuid}"
            rate = f[path].attrs.get("fps") if path in f else None
            if rate is None:
                rate = shared
            if rate is None:
                labelled = f" which {model} labels," if model is not None else ""
                raise ValueError(
                    f"{cpfile} records no frame rate for session {uuid},{labelled} so its syllable "
                    f"durations cannot be put in seconds. Compute the changepoints again, with the "
                    f"notebook's changepoints cell or `moseq2-pca compute-changepoints` in the project "
                    f"folder, which records each session's frame rate."
                )
            rates.append(float(rate))
    return rates


def parse_index(index: Union[str, dict]) -> tuple:
    """
    Load an index file, and use extraction UUIDs to sort index.

    Args:
    index (str or dict): if str, must be a path to the index file. If dict, must be the unsorted index.

    Returns:
    index (dict): loaded index file contents in a dictionary
    sorted_index (dict): index where the files have been sorted by UUID and pca_score path.
    """

    from_dict = True
    if isinstance(index, str):
        index_dir = dirname(index)
        index = moseq2_common.yaml.read_yaml(index)
        from_dict = False

    files = index["files"]

    sorted_index = groupby("uuid", files)
    # grab first entry in list, which is a dict
    sorted_index = valmap(first, sorted_index)
    # remove redundant uuid entry
    sorted_index = valmap(lambda d: dissoc(d, "uuid"), sorted_index)
    # tuple-ize the path entry, join with the index file dirname
    if not from_dict:
        sorted_index = valmap(
            lambda d: assoc(d, "path", tuple(join(index_dir, x) for x in d["path"])),
            sorted_index,
        )
    else:
        sorted_index = valmap(
            lambda d: assoc(d, "path", tuple(d["path"])), sorted_index
        )

    if index["pca_path"] is None:
        raise ValueError(
            "The index given has no pca_path, the PCA scores file holding its sessions' scores. Set pca_path "
            "to the scores file apply-pca wrote, or pass a project's sessions "
            "(moseq2_common.bookkeeping.session_index), which name the project's scores."
        )
    uuid_sorted = {
        "files": sorted_index,
        "pca_path": (
            join(index_dir, index["pca_path"]) if not from_dict else index["pca_path"]
        ),
    }

    return index, uuid_sorted


def get_sorted_index(index_file: str) -> dict:
    """
    Return the sorted index from an index_file path.

    Args:
    index_file (str): path to index file.

    Returns:
    sorted_ind (dict): dictionary of loaded sorted index file contents
    """

    _, sorted_ind = parse_index(index_file)
    return sorted_ind


def h5_filepath_from_sorted(sorted_index_entry: dict) -> str:
    """
    Get the h5 extraction file path from a sorted index entry

    Args:
    sorted_index_entry (dict): get filepath from sorted index.

    Returns:
    (str): a str containing the extraction filepath
    """

    return first(sorted_index_entry["path"])


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    Slice a numpy.array into subarrays with a given stride

    Args:
    a (numpy.array): array to get subarrays from.
    L (int): window length.
    S (int): stride size.

    Returns:
    (np.ndarray): sliced subarrays
    """

    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


@curry
def star(f, args):
    """
    Apply a function to a tuple of args, by expanding the tuple into each of the function's parameters.

    Args:
    f (function): a function that takes multiple arguments
    args (tuple): : a tuple to expand into ``f``

    Returns:
    the output of ``f``
    """

    return f(*args)


def assert_model_and_index_uuids_match(model, index):
    """
    Assert that both the model and index file contain the same set of UUIDs.

    Args:
    model (str or dict): str path to the model or dictionary of model.
    index (str or dict): str path to index file (moseq2-index.yaml) or dictionary of index data.
    """
    from moseq2_viz.model.util import parse_model_results

    if isinstance(model, str) and exists(model):
        # Load the model
        model = parse_model_results(model)
    if isinstance(index, str) and exists(index):
        # Read index file
        index = get_sorted_index(index)

    index_uuids = set(index["files"])
    model_uuids = set(model["metadata"]["uuids"])

    assert index_uuids == model_uuids, "Index file UUIDS must match the model UUIDs."

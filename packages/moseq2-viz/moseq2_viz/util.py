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
from moseq2_common.yaml import read_yaml as _read_yaml


def camel_to_snake(s):
    """
    Convert CamelCase to snake_case

    Args:
    s (str): string to convert to snake case

    Returns:
    (str): snake_case string
    """
    # https://gist.github.com/jaytaylor/3660565
    _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
    _underscorer2 = re.compile(r"([a-z0-9])([A-Z])")

    subbed = _underscorer1.sub(r"\1_\2", s)
    return _underscorer2.sub(r"\1_\2", subbed).lower()


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


TIMESTAMP_UNITS_IN_MS = {"milliseconds": 1, "ms": 1, "seconds": 1000, "s": 1000}


def read_timestamps(h5file: str):
    """
    Read an extraction's frame times in milliseconds, by the unit its timestamps
    dataset records. A file that records no unit (extracted before the unit was
    recorded) is returned as stored.

    Args:
    h5file (str): path to h5 file.

    Returns:
    timestamps (np.ndarray): the time of every frame, in milliseconds.
    unit_recorded (bool): whether the file records its timestamps' unit.
    """

    with h5py.File(h5file, "r") as f:
        # v0.1.3 or greater data format keeps them at /timestamps
        path = "timestamps" if "timestamps" in f else "metadata/timestamps"
        timestamps = f[path][()]
        unit = f[path].attrs.get("unit")
    if unit is None:
        return timestamps, False
    if isinstance(unit, bytes):
        unit = unit.decode()
    if unit not in TIMESTAMP_UNITS_IN_MS:
        raise ValueError(
            f"{h5file}: the timestamps at /{path} record their unit as {unit!r}, which is not "
            f"one of {', '.join(TIMESTAMP_UNITS_IN_MS)}. moseq2-extract writes milliseconds; "
            "check that this file is the extraction's own results h5 and is not damaged, or "
            "extract the session again."
        )
    factor = TIMESTAMP_UNITS_IN_MS[unit]
    return (timestamps if factor == 1 else timestamps * factor), True


def get_timestamps_from_h5(h5file: str) -> np.ndarray:
    """
    Return an extraction's frame times, in milliseconds when the file records
    their unit and as stored when it does not.

    Args:
    h5file (str): path to h5 file.

    Returns:
    (np.ndarray): timestamps from extraction within the h5file.
    """

    return read_timestamps(h5file)[0]


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


def load_timestamps(timestamp_file, col=0):
    """
    Read timestamps from space delimited text file.

    Args:
    timestamp_file (str): path to timestamp file
    col (int): column to load.

    Returns:
    ts (np.ndarray): loaded array of timestamps
    """

    ts = np.loadtxt(timestamp_file, delimiter=" ")
    if ts.ndim > 1:
        return ts[:, col]
    elif col > 0:
        raise Exception(
            f"Timestamp file {timestamp_file} does not have more than one column of data"
        )
    return ts


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
        index = read_yaml(index)
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
        raise Exception("Please add pac_score path to moseq2-index.yaml")
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


def recursive_find_h5s(root_dir=None, ext=".h5", yaml_string="{}.yaml"):
    """
    Recursively find h5 files, along with yaml files with the same basename.

    Args:
    root_dir (str): path to directory containing h5, if None, searches the current working directory
    ext (str): extension to search for.
    yaml_string (str): yaml file format name.

    Returns:
    h5s (list): list of paths to h5 files
    dicts (list): list of paths to metadata files
    yamls (list): list of paths to yaml files
    """

    if root_dir is None:
        root_dir = os.getcwd()

    def has_frames(h5f):
        """Checks if the supplied h5 file has a frames key"""
        with h5py.File(h5f, "r") as f:
            return "frames" in f

    def h5_to_yaml(h5f):
        return yaml_string.format(splitext(h5f)[0])

    # make function to test if yaml file with same basename as h5 file exists
    yaml_exists = compose(exists, h5_to_yaml)

    # grab all files with ext = .h5
    files = glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
    # keep h5s that have a yaml file associated with them
    to_keep = filter(yaml_exists, files)
    # keep h5s that have a frames key
    to_keep = filter(has_frames, to_keep)

    h5s = list(to_keep)
    yamls = list(map(h5_to_yaml, h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def read_yaml(yaml_path: str):
    """
    Read a given yaml file path into a dictionary object.

    Args:
    yaml_path (str): path to yaml file to read.

    Returns:
    loaded (dict): loaded yaml file contents.

    """
    return _read_yaml(yaml_path)


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
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

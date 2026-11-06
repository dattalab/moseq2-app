"""Reading ``.h5`` files into nested dicts, in one place.

Extraction results, PCA scores, changepoints and kappa scans are read through
``h5_to_dict``, which mirrors a group of the file as a dict of dicts with each
dataset's value at its leaf.

HDF5 stores a character set and a length for text, not a Python type, and
h5py 3 hands variable-length text back as the undecoded ``bytes``. The
packages' consumers treat those values as ``str``: a session's uuid, subject
name and acquisition timestamps are compared against strings read from the
YAML index, used as dictionary keys and DataFrame values, and written straight
into YAML, where ``b'...'`` and ``'...'`` do not match. So ``read_dataset``
decodes variable-length UTF-8 to ``str``, per element for an array of text.

Only variable-length UTF-8 is decoded. Fixed-width datasets (``|S5``, holding
the extraction version) and ASCII datasets were ``bytes`` under h5py 2 as
well, and their consumers treat them that way. The distinction survives only
in the dataset's dtype, which is why the test is ``check_string_dtype`` on the
dataset rather than ``isinstance`` on the value.

A dataset with a null dataspace, which is how an extraction records a
parameter left unset, is read directly and comes back as ``h5py.Empty``.
"""

from __future__ import annotations

import os

import h5py
import numpy as np

__all__ = ["extraction_labels", "h5_to_dict", "read_dataset"]


def read_dataset(dataset: h5py.Dataset):
    """One dataset's value, with variable-length UTF-8 text decoded to ``str``.

    Args:
    dataset (h5py.Dataset): the dataset to read.

    Returns:
    the dataset's contents; variable-length UTF-8 text as ``str`` (an object
    array of ``str`` for an array of text), anything else as h5py returns it.
    """
    info = h5py.check_string_dtype(dataset.dtype)
    if (info is None or info.length is not None or info.encoding != "utf-8"
            or dataset.shape is None):
        return dataset[()]
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        decoded = [v.decode("utf-8") if isinstance(v, bytes) else v for v in value.ravel()]
        return np.array(decoded, dtype=object).reshape(value.shape)
    return value


def _group_to_dict(file: h5py.File | h5py.Group, path: str) -> dict:
    node = file[path]
    if isinstance(node, h5py.Dataset):
        return {path.split("/")[-1]: read_dataset(node)}
    out = {}
    for key, item in node.items():
        if isinstance(item, h5py.Dataset):
            out[key] = read_dataset(item)
        elif isinstance(item, h5py.Group):
            out[key] = _group_to_dict(file, "/".join([path, key]))
    return out


def h5_to_dict(h5file: str | os.PathLike | h5py.File | h5py.Group, path: str = "/") -> dict:
    """The contents of an ``.h5`` file below ``path``, as nested dicts.

    Args:
    h5file (str, os.PathLike, h5py.File or h5py.Group): the file's path, or an
        open file or group, which is left open.
    path (str): the group to read, relative to ``h5file`` (or absolute). A path
        that names a dataset returns ``{name: value}``. Default: the whole file.

    Returns:
    a dict with one entry per dataset and one nested dict per group.
    """
    if isinstance(h5file, (h5py.File, h5py.Group)):
        return _group_to_dict(h5file, path)
    if isinstance(h5file, (str, os.PathLike)):
        with h5py.File(h5file, "r") as f:
            return _group_to_dict(f, path)
    raise TypeError(
        f"h5_to_dict was given {h5file!r} (a {type(h5file).__name__}); pass the path "
        "of an .h5 file or an open h5py.File or h5py.Group."
    )


def _label(value):
    """An acquisition value as metadata.json held it: text as ``str``, arrays as lists, unset as None."""
    if isinstance(value, h5py.Empty):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return [_label(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def extraction_labels(h5file: str | os.PathLike) -> tuple[str | None, dict]:
    """An extraction's session uuid (``metadata/uuid``) and acquisition metadata (``metadata/acquisition``,
    the recording's metadata.json as extraction read it), from its ``.h5``."""
    with h5py.File(h5file, "r") as f:
        uuid = _label(read_dataset(f["metadata/uuid"])) if "metadata/uuid" in f else None
        group = f["metadata/acquisition"] if "metadata/acquisition" in f else {}
        metadata = {name: _label(read_dataset(node)) for name, node in group.items()
                    if isinstance(node, h5py.Dataset)}
    return (str(uuid) if uuid is not None else None), metadata

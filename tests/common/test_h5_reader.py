"""Every package reads ``.h5`` files through one reader in moseq2-common.

Extraction results, PCA scores, changepoints and model exports are read into
nested dicts by ``moseq2_common.h5.h5_to_dict``. Before it, extract, pca and
viz each kept their own copy, each with its own fix for h5py 3 handing
variable-length text back as ``bytes``, so the next h5py or file-format change
would have been fixed in one copy and missed in the others.

What the reader returns, by how a value is stored:

- variable-length UTF-8 text (a session's uuid, subject name, acquisition
  timestamps) reads as ``str``, alone or as an object array of ``str``, so it
  matches the strings read from the YAML index;
- fixed-width text (``|S5``, as ``extract_version`` is stored) stays ``bytes``,
  as it always was;
- a null dataspace, which is how an extraction records an unset parameter,
  reads as ``h5py.Empty``.

moseq2-model keeps its own copy until its decoding is switched, which changes
what a model saved as ``.h5`` reads back.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from moseq2_common.h5 import h5_to_dict, read_dataset

PACKAGES = ["moseq2_extract", "moseq2_pca", "moseq2_model", "moseq2_viz", "moseq2_app"]
READERS = {"h5_to_dict", "_load_h5_to_dict", "_read_dataset", "read_dataset"}


@pytest.fixture
def h5file(tmp_path):
    path = tmp_path / "session.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata/uuid", data="5c72bf30-9596-4d4d-ae38-db9a7a28e912",
                         dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("metadata/names", data=["mouse 1", "mouse 2"],
                         dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("metadata/extract_version", data=np.bytes_("1.5.0"))
        f.create_dataset("metadata/extraction/parameters/bg_roi_index",
                         data=h5py.Empty("f"))
        f.create_dataset("scalars/centroid_x_px", data=np.arange(4, dtype="float32"))
        f.create_dataset("scalars/velocity_2d_mm", data=np.ones(4))
    return path


def test_variable_length_utf8_text_reads_as_str(h5file):
    with h5py.File(h5file, "r") as f:
        value = read_dataset(f["metadata/uuid"])
    assert value == "5c72bf30-9596-4d4d-ae38-db9a7a28e912"
    assert type(value) is str


def test_an_array_of_text_reads_as_an_object_array_of_str(h5file):
    with h5py.File(h5file, "r") as f:
        value = read_dataset(f["metadata/names"])
    assert value.dtype == object
    assert value.tolist() == ["mouse 1", "mouse 2"]
    assert all(type(v) is str for v in value)


def test_fixed_width_text_stays_bytes(h5file):
    with h5py.File(h5file, "r") as f:
        assert f["metadata/extract_version"].dtype == np.dtype("|S5")
        value = read_dataset(f["metadata/extract_version"])
    assert value == b"1.5.0"


def test_an_unset_parameter_reads_as_empty(h5file):
    with h5py.File(h5file, "r") as f:
        value = read_dataset(f["metadata/extraction/parameters/bg_roi_index"])
    assert isinstance(value, h5py.Empty)


def test_a_dataset_path_reads_as_a_one_entry_dict(h5file):
    out = h5_to_dict(h5file, "scalars/centroid_x_px")
    assert list(out) == ["centroid_x_px"]
    np.testing.assert_array_equal(out["centroid_x_px"], np.arange(4, dtype="float32"))


def test_an_open_group_is_accepted(h5file):
    with h5py.File(h5file, "r") as f:
        out = h5_to_dict(f["metadata"], "extraction")
    assert set(out) == {"parameters"}
    assert isinstance(out["parameters"]["bg_roi_index"], h5py.Empty)


def test_nested_groups_read_as_nested_dicts(h5file):
    for source in (str(h5file), Path(h5file)):
        out = h5_to_dict(source)
        assert set(out) == {"metadata", "scalars"}
        assert set(out["scalars"]) == {"centroid_x_px", "velocity_2d_mm"}
        assert out["metadata"]["uuid"] == "5c72bf30-9596-4d4d-ae38-db9a7a28e912"
        assert out["metadata"]["extract_version"] == b"1.5.0"
        assert isinstance(out["metadata"]["extraction"]["parameters"]["bg_roi_index"],
                          h5py.Empty)


def test_only_moseq2_common_and_moseq2_model_define_an_h5_reader():
    found = []
    for pkg in PACKAGES:
        root = Path(importlib.import_module(pkg).__file__).parent
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module = path.relative_to(root.parent).as_posix()
            if module == "moseq2_model/util.py":
                continue  # its copy decodes no strings; switching it changes results
            found += [
                f"{module}:{node.lineno} {node.name}"
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in READERS
            ]
    assert found == []

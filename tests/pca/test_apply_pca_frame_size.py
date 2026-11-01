"""apply-pca stops when the sessions' frames are not the size the PCA was trained on.

Each trained component is one value per pixel of a frame, so a PCA trained on
80 x 80 extractions can only score 80 x 80 frames. Applied to frames of
another size (extractions made with another ``crop_size``), the projection
fails deep in dask with a shape error that names no session. The check names
each session that does not match, with its file, UUID and frame size, and the
size the PCA expects.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import moseq2_pca.helpers.wrappers as wrappers
from moseq2_pca.cli import apply_pca
from moseq2_pca.util import write_yaml


class ReachedDask(Exception):
    """Raised in place of starting dask: the run got past every check."""


def make_sessions(root, shapes):
    paths = {}
    for uuid, shape in shapes.items():
        folder = root / uuid
        folder.mkdir(parents=True)
        h5 = folder / "results_00.h5"
        with h5py.File(h5, "w") as f:
            f["frames"] = np.zeros((5, *shape), dtype="uint8")
            f["timestamps"] = np.arange(5) * 33.3
        write_yaml({"uuid": uuid}, str(folder / "results_00.yaml"))
        paths[uuid] = str(h5)
    return paths


def make_pca(folder, pixels):
    folder.mkdir(parents=True, exist_ok=True)
    pca_file = folder / "pca.h5"
    with h5py.File(pca_file, "w") as f:
        f["components"] = np.zeros((3, pixels), dtype="float32")
        f["mean"] = np.zeros(pixels, dtype="float32")
    write_yaml({"use_fft": False, "missing_data": False, "tailfilter_shape": "ellipse",
                "tailfilter_size": [9, 9], "gaussfilter_space": [1.5, 1], "gaussfilter_time": 0,
                "medfilter_time": [0], "medfilter_space": [0], "mask_height_threshold": 5,
                "mask_threshold": -16, "min_height": 10, "max_height": 120},
               str(folder / "pca.yaml"))
    return str(pca_file)


def config(pca_file):
    """apply-pca's defaults, as the notebook passes them, with this PCA."""
    defaults = {p.name: p.default for p in apply_pca.params if not p.required}
    return {**defaults, "cluster_type": "local", "h5_path": "/frames", "pca_path": "/components",
            "pca_file": pca_file, "overwrite_pca_apply": True, "missing_data": False,
            "use_fft": False, "dask_cache_path": str(pca_file) + "_cache"}


@pytest.fixture
def no_dask(monkeypatch):
    def stop(*args, **kwargs):
        raise ReachedDask

    monkeypatch.setattr(wrappers, "initialize_dask", stop)
    monkeypatch.setattr(wrappers, "apply_pca_local", stop)


def test_sessions_of_another_size_stop_before_scoring(tmp_path, no_dask):
    """Every session is 64 x 64 and the PCA was trained on 80 x 80 frames."""
    paths = make_sessions(tmp_path / "input", {"uuid-a": (64, 64), "uuid-b": (64, 64)})
    pca_file = make_pca(tmp_path / "_pca", 80 * 80)
    with pytest.raises(ValueError) as e:
        wrappers.apply_pca_wrapper(str(tmp_path / "input"), config(pca_file),
                                   str(tmp_path / "_pca"), "pca_scores")
    message = str(e.value)
    for uuid, h5 in paths.items():
        assert uuid in message and h5 in message
    assert "64 x 64" in message
    assert "6400" in message and "80 x 80" in message
    assert pca_file in message
    assert "crop_size" in message


def test_only_the_mismatching_sessions_are_listed(tmp_path, no_dask, monkeypatch):
    """With the across-sessions check out of the way, one odd session is the one named."""
    monkeypatch.setattr(wrappers, "check_frame_shapes", lambda *a, **k: None)
    paths = make_sessions(tmp_path / "input", {"uuid-ok": (80, 80), "uuid-odd": (90, 90)})
    pca_file = make_pca(tmp_path / "_pca", 80 * 80)
    with pytest.raises(ValueError) as e:
        wrappers.apply_pca_wrapper(str(tmp_path / "input"), config(pca_file),
                                   str(tmp_path / "_pca"), "pca_scores")
    message = str(e.value)
    assert "uuid-odd" in message and paths["uuid-odd"] in message and "90 x 90" in message
    assert "uuid-ok" not in message


def test_matching_sessions_go_on_to_scoring(tmp_path, no_dask):
    make_sessions(tmp_path / "input", {"uuid-a": (80, 80), "uuid-b": (80, 80)})
    pca_file = make_pca(tmp_path / "_pca", 80 * 80)
    with pytest.raises(ReachedDask):
        wrappers.apply_pca_wrapper(str(tmp_path / "input"), config(pca_file),
                                   str(tmp_path / "_pca"), "pca_scores")

"""PCA stops before it starts when sessions were extracted at different crop sizes.

Every session's frames go into one matrix whose columns are the pixels of a
frame, so frames of different sizes cannot share one set of components.
Without a check, mixing extractions made with different ``crop_size`` values
fails deep in dask with "Shapes do not align" (moseq2-pca#44). The check
names every session, its file and UUID, with its frame shape.
"""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

from moseq2_pca.helpers.wrappers import (apply_pca_wrapper, compute_changepoints_wrapper,
                                         train_pca_wrapper)
from moseq2_pca.util import write_yaml

SHAPES = {"uuid-80": (80, 80), "uuid-also-80": (80, 80), "uuid-120": (120, 120)}


@pytest.fixture
def sessions(tmp_path):
    """Three extractions, one of them at a different crop size."""
    paths = {}
    for uuid, shape in SHAPES.items():
        folder = tmp_path / "input" / uuid
        folder.mkdir(parents=True)
        h5 = folder / "results_00.h5"
        with h5py.File(h5, "w") as f:
            f["frames"] = np.zeros((5, *shape), dtype="uint8")
            f["timestamps"] = np.arange(5) * 33.3
        with h5py.File(folder / "results_00.h5", "a") as f:
            f["metadata/uuid"] = uuid
        paths[uuid] = str(h5)
    return tmp_path / "input", paths


CONFIG = {"cluster_type": "local", "h5_path": "/frames", "missing_data": False, "use_fft": False}


@pytest.mark.parametrize(
    "wrapper", [train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper],
    ids=["train", "apply", "changepoints"],
)
def test_mixed_crop_sizes_stop_before_pca(tmp_path, sessions, wrapper):
    input_dir, paths = sessions
    with pytest.raises(ValueError) as e:
        wrapper(str(input_dir), dict(CONFIG), str(tmp_path / "_pca"), "out")
    message = str(e.value)
    for uuid, h5 in paths.items():
        assert uuid in message and h5 in message
    assert "80 x 80" in message and "120 x 120" in message
    assert "crop_size" in message


def test_one_crop_size_passes_the_check(tmp_path, sessions, monkeypatch):
    """With the odd session removed, training gets past the check to the overwrite prompt."""
    input_dir, _ = sessions
    shutil.rmtree(input_dir / "uuid-120")
    (tmp_path / "_pca").mkdir()
    (tmp_path / "_pca" / "out.h5").touch()
    asked = []
    monkeypatch.setattr("builtins.input", lambda *a: asked.append(True) or "n")
    train_pca_wrapper(str(input_dir), dict(CONFIG), str(tmp_path / "_pca"), "out")
    assert asked

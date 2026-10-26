"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


@pytest.fixture
def synthetic_flip_session(tmp_path, monkeypatch):
    from moseq2_app.flip import widget
    path = tmp_path / "synthetic.h5"
    frames = np.zeros((12, 8, 8), dtype=np.uint8)
    frames[:, 1:3, 2:5] = np.arange(1, 13)[:, None, None]
    with h5py.File(path, "w") as f:
        f["frames"] = frames
        f["scalars/angle"] = np.zeros(12)
        f.create_group("metadata/extraction")
    monkeypatch.setattr(widget, "get_flips", lambda values, **kw: np.ones(len(values), dtype=bool))
    subject = SimpleNamespace(sessions={"synthetic": path})
    return widget, subject, path, frames


def test_flip_updates_the_angles_of_the_matching_batch(synthetic_flip_session):
    widget, subject, path, original = synthetic_flip_session
    widget.FlipClassifierWidget.apply_flip_classifier(subject, "synthetic-unused", chunk_size=4,
                                                       smoothing=1, verbose=False)
    with h5py.File(path) as f:
        np.testing.assert_allclose(f["scalars/angle"][:11], np.pi)
        np.testing.assert_array_equal(f["frames"][:11], np.rot90(original[:11], 2, axes=(1, 2)))

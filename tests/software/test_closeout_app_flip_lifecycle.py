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


def test_flip_refuses_reprocessing_a_completed_session(synthetic_flip_session):
    widget, subject, path, _ = synthetic_flip_session
    for i in range(2):
        widget.FlipClassifierWidget.apply_flip_classifier(subject, "synthetic-unused",
                                                           chunk_size=20, smoothing=1, verbose=False)
        with h5py.File(path) as f:
            current = f["frames"][:]
            if i == 0:
                first = current
            else:
                np.testing.assert_array_equal(current, first)


def test_interrupted_flip_refuses_mutating_rerun(synthetic_flip_session, monkeypatch):
    widget, subject, path, _ = synthetic_flip_session
    calls = 0
    def interrupt(values, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected synthetic interruption")
        return np.ones(len(values), dtype=bool)
    monkeypatch.setattr(widget, "get_flips", interrupt)
    with pytest.raises(RuntimeError, match="injected"):
        widget.FlipClassifierWidget.apply_flip_classifier(subject, "synthetic-unused",
                                                           chunk_size=4, smoothing=1, verbose=False)
    with h5py.File(path) as f:
        after_failure = f["frames"][:]
    monkeypatch.setattr(widget, "get_flips", lambda values, **kw: np.ones(len(values), dtype=bool))
    with pytest.raises(RuntimeError, match="(?i)(incomplete|failed|restore|interrupted)"):
        widget.FlipClassifierWidget.apply_flip_classifier(subject, "synthetic-unused",
                                                           chunk_size=4, smoothing=1, verbose=False)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["frames"][:], after_failure)

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


def test_preview_pipe_is_reaped_when_processing_raises(synthetic_flip_session, monkeypatch):
    widget, subject, _, _ = synthetic_flip_session
    pipe = SimpleNamespace(returncode=0, reaped=False)
    def reap():
        pipe.reaped = True
    pipe.communicate = reap
    monkeypatch.setattr(widget, "write_frames_preview", lambda *a, **kw: pipe)
    calls = 0
    def interrupt(values, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic processing failure")
        return np.ones(len(values), dtype=bool)
    monkeypatch.setattr(widget, "get_flips", interrupt)
    with pytest.raises(RuntimeError, match="synthetic processing"):
        widget.FlipClassifierWidget.apply_flip_classifier(subject, "synthetic-unused", chunk_size=4,
                                                           smoothing=1, verbose=False, write_movie=True)
    assert pipe.reaped

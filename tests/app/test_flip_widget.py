"""Correcting extractions with a trained flip classifier.

The Flip-Classifier-Training notebook ends by calling
``FlipClassifierWidget.apply_flip_classifier``, which walks each extraction in
chunks, asks the classifier which frames face the wrong way, rotates those
frames by 180 degrees in place and adds pi to their recorded angle. The
suite walks neither the notebook nor the widget, so it is tested here
directly. The classifier itself is replaced by one that flags every frame,
which makes the correction a plain function of the frame index.
"""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from moseq2_app.flip import widget
from video import count_frames

NFRAMES = 12


@pytest.fixture
def session(tmp_path, monkeypatch):
    """One extraction whose every frame is different, with zero angles."""
    path = tmp_path / "session.h5"
    frames = np.zeros((NFRAMES, 8, 8), dtype=np.uint8)
    frames[:, 1:3, 2:5] = np.arange(1, NFRAMES + 1)[:, None, None]
    with h5py.File(path, "w") as f:
        f["frames"] = frames
        f["scalars/angle"] = np.zeros(NFRAMES)
        f.create_group("metadata/extraction")
    monkeypatch.setattr(widget, "get_flips", lambda values, **kw: np.ones(len(values), dtype=bool))
    return SimpleNamespace(sessions={"session": path}), path, frames


def correct(subject, **kwargs):
    widget.FlipClassifierWidget.apply_flip_classifier(
        subject, "unused-classifier", smoothing=1, verbose=False, **kwargs
    )


def test_the_last_frame_is_corrected_like_the_rest(session):
    """The chunks cover the whole recording, the final frame included."""
    subject, path, original = session
    correct(subject, chunk_size=20)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["frames"][-1], np.rot90(original[-1], 2))
        assert f["scalars/angle"][-1] == pytest.approx(np.pi)


def test_each_chunk_corrects_its_own_angles(session):
    """A flagged frame's angle moves by pi once, whichever chunk it falls in."""
    subject, path, _ = session
    correct(subject, chunk_size=4)
    with h5py.File(path) as f:
        np.testing.assert_allclose(f["scalars/angle"][()], np.pi)


def test_overlapping_chunks_correct_each_frame_once(session):
    """A frame shared by two chunks is rotated and its angle moved only once."""
    subject, path, original = session
    correct(subject, chunk_size=5, chunk_overlap=2)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["frames"][()], np.rot90(original, 2, axes=(1, 2)))
        np.testing.assert_allclose(f["scalars/angle"][()], np.pi)


def test_overlapping_chunks_write_each_frame_to_the_movie_once(session):
    """The flipped movie holds one frame per extracted frame."""
    subject, path, _ = session
    correct(subject, chunk_size=5, chunk_overlap=2, write_movie=True)
    assert count_frames(str(path).replace(".h5", "_flipped.mp4")) == NFRAMES

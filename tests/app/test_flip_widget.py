"""Correcting extractions with a trained flip classifier.

The Flip-Classifier-Training notebook ends by calling
``FlipClassifierWidget.apply_flip_classifier``, which walks each extraction in
chunks, asks the classifier which frames face the wrong way, rotates those
frames by 180 degrees in place and adds pi to their recorded angle. Because
the change is made in place, each extraction records in its own file that
the correction ran, and a later run skips it. The suite walks neither the
notebook nor the widget, so it is tested here directly. The classifier itself is replaced by one that flags every frame,
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


def test_a_second_run_skips_a_corrected_session(session, capsys):
    """Running the cell again leaves corrected frames as they are."""
    subject, path, original = session
    correct(subject, chunk_size=4)
    correct(subject, chunk_size=4)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["frames"][()], np.rot90(original, 2, axes=(1, 2)))
        np.testing.assert_allclose(f["scalars/angle"][()], np.pi)
    assert "already applied to session; skipping it" in capsys.readouterr().out


def test_a_run_that_did_not_finish_is_refused(session, monkeypatch):
    """A session left half corrected is not touched again."""
    subject, path, _ = session
    calls = 0

    def fails_on_the_second_chunk(values, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("classifier stopped")
        return np.ones(len(values), dtype=bool)

    monkeypatch.setattr(widget, "get_flips", fails_on_the_second_chunk)
    with pytest.raises(RuntimeError, match="classifier stopped"):
        correct(subject, chunk_size=4)
    with h5py.File(path) as f:
        half_corrected = f["frames"][()]

    monkeypatch.setattr(widget, "get_flips", lambda values, **kw: np.ones(len(values), dtype=bool))
    with pytest.raises(RuntimeError, match="did not finish.*Re-extract this session"):
        correct(subject, chunk_size=4)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["frames"][()], half_corrected)


def test_a_movie_that_cannot_be_written_does_not_stop_the_correction(session, monkeypatch):
    """Every frame is still corrected when the preview movie fails."""
    subject, path, original = session

    def no_movie(*args, **kwargs):
        raise AttributeError("no writer")

    monkeypatch.setattr(widget, "write_frames_preview", no_movie)
    correct(subject, chunk_size=4, write_movie=True)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["frames"][()], np.rot90(original, 2, axes=(1, 2)))

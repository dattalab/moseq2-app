"""Building the flip classifier's training set.

The Flip-Classifier-Training notebook hands the frame ranges a user labeled
to ``create_training_dataset``, which assembles them into one array, reports
its shape and size, and saves it for training. The suite walks neither that
notebook nor this module, so the function is tested here directly.
"""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest

from moseq2_app.flip import train


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_the_reported_size_is_the_size_of_the_training_set(tmp_path, monkeypatch, capsys, dtype):
    """A user reads this number to decide whether training will fit in memory.

    It is the size of the array that gets saved, whatever its dtype: with two
    bytes a pixel it is twice the number of pixels.
    """
    session, index = tmp_path / "session.h5", tmp_path / "flip-training-frame-ranges.p"
    with h5py.File(session, "w") as f:
        f["frames"] = np.zeros((500, 80, 80), dtype=dtype)
    with index.open("wb") as f:
        pickle.dump(({"session": str(session)}, {"session": [(False, slice(None))]}), f)
    monkeypatch.setattr(train, "clean_frames", lambda frames, *a, **kw: frames)
    saved = {}
    monkeypatch.setattr(np, "savez", lambda path, **arrays: saved.update(arrays))

    train.create_training_dataset(str(index))

    assert f"memory usage: {saved['frames'].nbytes / 1e9:0.2f} GB" in capsys.readouterr().out


def test_training_cross_validates_and_fits(tmp_path):
    """The notebook's training cell runs to a classifier.

    ``train_classifier`` fits a PCA on the training set's cleaned third, then
    scores the classifier by four-fold cross-validation, which fits a copy of
    the pipeline per fold. Each copy has to carry the fitted PCA.
    """
    rng = np.random.default_rng(0)
    frames = np.zeros((120, 20, 20), dtype=np.uint8)
    frames[:, 8:12, 2:10] = 50
    frames[:, 6:14, 10:16] = 80
    frames += rng.integers(0, 5, size=frames.shape, dtype=np.uint8)
    frames[60:] = np.rot90(frames[60:], 2, axes=(1, 2))
    flipped = np.repeat([0, 1], 60).astype(np.uint8)
    data = tmp_path / "training_data.npz"
    np.savez(data, frames=frames, flipped=flipped)

    classifier = train.train_classifier(str(data), classifier="RF", n_components=5)

    np.testing.assert_array_equal(classifier.predict(frames), flipped)

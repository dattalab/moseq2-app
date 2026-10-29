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

from moseq2_app.flip import train


def test_the_reported_size_is_the_size_of_the_training_set(tmp_path, monkeypatch, capsys):
    """A user reads this number to decide whether training will fit in memory.

    It is the size of the array that gets saved, whatever its dtype.
    """
    session, index = tmp_path / "session.h5", tmp_path / "flip-training-frame-ranges.p"
    with h5py.File(session, "w") as f:
        f["frames"] = np.zeros((500, 80, 80), dtype=np.uint8)
    with index.open("wb") as f:
        pickle.dump(({"session": str(session)}, {"session": [(False, slice(None))]}), f)
    monkeypatch.setattr(train, "clean_frames", lambda frames, *a, **kw: frames)
    saved = {}
    monkeypatch.setattr(np, "savez", lambda path, **arrays: saved.update(arrays))

    train.create_training_dataset(str(index))

    assert f"memory usage: {saved['frames'].nbytes / 1e9:0.2f} GB" in capsys.readouterr().out

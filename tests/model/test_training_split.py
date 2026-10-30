"""Holding out the end of each session for validation with ``--percent-split``.

``learn-model --percent-split N`` keeps the last N% of every session's frames
out of training and tracks the model's likelihood on them. The rest of the
session, everything before that, is the training data, so each frame is used
exactly once. This is what sklearn's ``train_test_split(test_size=N / 100,
shuffle=False)`` did before the package stopped depending on it.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from moseq2_model.helpers.data import get_training_data_splits

FRAMES = 1000


def session():
    return OrderedDict([("session", np.arange(FRAMES)[:, None])])


@pytest.mark.parametrize("fraction", [0.2, 0.8])
def test_every_frame_is_used_once(fraction):
    """Training then validation reassemble the session, with no gap or overlap."""
    data = session()
    train, validation = get_training_data_splits(fraction, data)
    np.testing.assert_array_equal(
        np.concatenate([train["session"], validation["session"]]), data["session"]
    )


def test_validation_is_the_last_part_of_each_session():
    """--percent-split 20 validates on the last 20% and trains on the first 80%."""
    data = session()
    train, validation = get_training_data_splits(0.2, data)
    np.testing.assert_array_equal(validation["session"], data["session"][800:])
    np.testing.assert_array_equal(train["session"], data["session"][:800])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_a_fraction_outside_zero_to_one_is_refused(fraction):
    """There is no split that holds out none or all of a session."""
    with pytest.raises(ValueError):
        get_training_data_splits(fraction, session())

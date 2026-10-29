"""Whitening the PC scores before fitting, when part of the data is held out.

With ``--whiten all``, learn-model subtracts a mean and divides out a
covariance before fitting. Those are estimated from the frames the model
trains on: with ``--hold-out`` the training sessions, with ``--percent-split``
the part of each session before its validation frames. The held-out data is
whitened with the same transform but does not shape it, and the transform
saved in the model is the one apply-model later uses on new data.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from moseq2_model.helpers.data import prepare_model_metadata
from moseq2_model.train.util import whiten_all

CONFIG = dict(kappa=100, hold_out=False, nfolds=2, hold_out_seed=0, ncpus=1, gamma=1, alpha=1, nlags=1,
              robust=False, max_states=2, separate_trans=False, whiten="all", noise_level=0, percent_split=0)


def prepare(data, **config):
    groups = {"groups": dict.fromkeys(data, "g")}
    return prepare_model_metadata(OrderedDict(data), groups, {**CONFIG, **config})


def two_sessions(offset):
    rng = np.random.RandomState(17)
    return {"a": rng.normal(size=(40, 2)), "b": rng.normal(size=(40, 2)) + offset}


def test_a_held_out_session_does_not_shape_the_training_whitening():
    first, second = prepare(two_sessions(100), hold_out=True), prepare(two_sessions(1000), hold_out=True)
    assert first[2] == second[2] == ["a"] and first[3] == second[3] == ["b"]
    np.testing.assert_array_equal(first[0]["a"], second[0]["a"])


def test_validation_frames_do_not_shape_the_training_whitening():
    def one_session(offset):
        x = np.random.RandomState(17).normal(size=(40, 2))
        x[20:] += offset           # the last half is the validation part with --percent-split 50
        return {"a": x}
    first, second = prepare(one_session(100), percent_split=50), prepare(one_session(1000), percent_split=50)
    np.testing.assert_array_equal(first[0]["a"][:20], second[0]["a"][:20])


def test_without_held_out_data_the_whitening_uses_everything():
    data = two_sessions(3)
    whitened, *_ = prepare(data)
    expected, _ = whiten_all(OrderedDict(data))
    for key in data:
        np.testing.assert_array_equal(whitened[key], expected[key])

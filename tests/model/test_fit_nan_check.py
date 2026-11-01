"""A fit whose model goes NaN stops with an error and saves no model.

pyhsmm raised as soon as a fit blew up numerically. JAX does not: a NaN in
one sweep's parameters spreads through every later sweep, and the fit would
finish and save a model full of NaNs. ``train_model`` therefore checks the
model after every Gibbs sweep with ``jax_moseq.utils.check_for_nans``, the
check keypoint-moseq runs, and stops with ``ModelNaNError``: which arrays hold
NaNs and how many, the sweep, the sessions involved where an array is per
session, the likely causes in the data and what to do. ``learn-model`` writes
no model file, and checkpoints already written stay, the last one named as
the last state without NaNs.

The NaNs are put into the model after a chosen sweep, by wrapping the
backend's ``resample_model``, on a small synthetic fit.
"""

from __future__ import annotations

import re

import h5py
import jax.numpy as jnp
import joblib
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_model.train.util as train_util
from moseq2_model.cli import cli
from moseq2_model.train.backend import JaxARHMM
from moseq2_model.train.util import ModelNaNError, train_model

SESSIONS = {"s0": 120, "s1": 140, "s2": 160}
NUM_ITER = 6


@pytest.fixture
def scores(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "pca_scores.h5"
    with h5py.File(path, "w") as f:
        for key, n in SESSIONS.items():
            f[f"scores/{key}"] = rng.normal(size=(n, 10)).astype("float32")
            f[f"scores_idx/{key}"] = np.arange(n, dtype="float32")
    return path


@pytest.fixture
def sweeps(monkeypatch):
    """Count sweeps, and after sweep ``at`` apply ``poison`` to the model dict."""
    record = {"count": 0, "at": None, "poison": None}
    original = JaxARHMM.resample_model

    def resample_model(self):
        original(self)
        record["count"] += 1
        if record["count"] == record["at"]:
            record["poison"](self._model)

    monkeypatch.setattr(JaxARHMM, "resample_model", resample_model)
    return record


def nan_in_ab(model):
    """Twelve NaNs in state 0's AR matrix."""
    model["params"]["Ab"] = model["params"]["Ab"].at[0, :2, :6].set(jnp.nan)


def learn(scores, *args):
    model = scores.parent / "model.p"
    return model, CliRunner().invoke(cli, [
        "learn-model", str(scores), str(model), "-n", str(NUM_ITER),
        "-p", "False", "--max-states", "5", "--nlags", "1", *args,
    ], catch_exceptions=False)


def test_nans_stop_the_fit_and_no_model_is_written(scores, sweeps):
    sweeps.update(at=3, poison=nan_in_ab)
    with pytest.raises(ModelNaNError) as raised:
        learn(scores)
    message = str(raised.value)
    assert "12 NaNs found in params/Ab" in message
    assert "after sweep 3 of 6" in message
    assert sweeps["count"] == 3
    assert not (scores.parent / "model.p").exists()


def test_the_message_names_the_causes_and_what_to_do(scores, sweeps):
    sweeps.update(at=2, poison=nan_in_ab)
    with pytest.raises(ModelNaNError) as raised:
        learn(scores)
    message = str(raised.value)
    assert "No model file was written" in message
    assert str(scores.parent / "model.p") in message
    assert "missing from the frame or still" in message
    assert "background" in message
    assert "leave out" in message and "fit again" in message
    # the notebook's cell, by its heading, and the option that makes learn-model read the index
    assert "QC cell" not in message
    assert "Run Extraction Validation Tests" in message
    assert "only when given -i" in message
    assert "No checkpoint of this fit was saved" in message


def test_the_last_checkpoint_is_kept_and_named(scores, sweeps):
    sweeps.update(at=5, poison=nan_in_ab)
    with pytest.raises(ModelNaNError) as raised:
        learn(scores, "--checkpoint-freq", "2")
    checkpoints = scores.parent / "checkpoints"
    kept = sorted(p.name for p in checkpoints.iterdir())
    assert kept == ["model-checkpoint_1.arhmm", "model-checkpoint_3.arhmm"]
    message = str(raised.value)
    assert str(checkpoints / "model-checkpoint_3.arhmm") in message
    assert "after sweep 4" in message
    assert not (scores.parent / "model.p").exists()


def test_a_resumed_fit_names_the_checkpoint_it_resumed_from(scores, sweeps):
    learn(scores, "--checkpoint-freq", "2", "-n", "2")
    resumed = scores.parent / "checkpoints" / "model-checkpoint_1.arhmm"
    (scores.parent / "model.p").unlink()
    sweeps.update(count=0, at=1, poison=nan_in_ab)
    with pytest.raises(ModelNaNError) as raised:
        learn(scores, "--checkpoint-freq", "2", "--use-checkpoint")
    message = str(raised.value)
    assert "after sweep 3 of 6" in message
    assert str(resumed) in message


def test_per_session_arrays_name_their_sessions(scores, sweeps):
    def nan_in_tau(model):
        tau = model["states"]["tau"]
        model["states"]["tau"] = tau.at[1, :7].set(jnp.nan)

    sweeps.update(at=1, poison=nan_in_tau)
    with pytest.raises(ModelNaNError) as raised:
        learn(scores, "--robust")
    message = str(raised.value)
    assert "7 NaNs found in states/tau" in message
    assert re.search(r"states/tau.*\bs1\b", message, re.S)
    assert "s0" not in message.split("states/tau", 1)[1].split("\n\n", 1)[0]


def test_a_healthy_fit_runs_every_sweep_and_saves(scores, sweeps):
    model, result = learn(scores)
    assert result.exit_code == 0, result.output
    assert sweeps["count"] == NUM_ITER
    saved = joblib.load(model)
    assert np.isfinite(saved["model_parameters"]["ar_mat"]).all()


class Bar:
    """A progress bar that records being closed."""

    def __init__(self, iterable, **kwargs):
        self.iterable, self.closed = iterable, False
        Bar.last = self

    def __iter__(self):
        return iter(self.iterable)

    def close(self):
        self.closed = True


class Poisoned:
    """A model whose state is NaN from the first sweep."""

    nlags = 1
    states_list = ()
    _session_order = ["only"]

    def __init__(self):
        self.model_dict = {"params": {"Ab": jnp.zeros(3)}}

    def resample_model(self):
        self.model_dict = {"params": {"Ab": jnp.full(3, jnp.nan)}}

    def log_likelihood(self):
        return 0.0


def test_the_progress_bar_is_closed(monkeypatch):
    monkeypatch.setattr(train_util, "tqdm", Bar)
    with pytest.raises(ModelNaNError, match="3 NaNs found in params/Ab"):
        train_model(Poisoned(), num_iter=2)
    assert Bar.last.closed

"""Which session each exported label sequence belongs to.

``learn-model`` saves ``labels``, one syllable sequence per modeled session,
and ``keys``, the sessions they belong to, in the same order. With
``--hold-out`` a fold of sessions is set aside and the rest are trained in
shuffled order, so ``keys`` has to follow the order the model trained in for
``zip(keys, labels)`` to pair each session with its own syllables. Held-out
sessions have no labels and are listed in ``hold_out_list`` instead.

The sessions here have different lengths, so a label sequence's length says
which session it came from.
"""

from __future__ import annotations

import h5py
import joblib
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_model.cli import cli

LENGTHS = {"s0": 120, "s1": 140, "s2": 160, "s3": 180, "s4": 200}


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A one-iteration fit of five sessions with one held out."""
    folder = tmp_path_factory.mktemp("fit")
    rng = np.random.default_rng(0)
    with h5py.File(folder / "pca_scores.h5", "w") as f:
        for key, n in LENGTHS.items():
            f[f"scores/{key}"] = rng.normal(size=(n, 10)).astype("float32")
            f[f"scores_idx/{key}"] = np.arange(n, dtype="float32")
    result = CliRunner().invoke(cli, [
        "learn-model", str(folder / "pca_scores.h5"), str(folder / "model.p"),
        "--hold-out", "--hold-out-seed", "0", "-n", "1",
        "--save-model", "False", "-p", "False", "-c", "1",
    ])
    assert result.exit_code == 0, result.output
    return joblib.load(folder / "model.p")


def test_each_key_names_the_session_its_labels_came_from(fitted):
    """zip(keys, labels) gives every session its own syllables."""
    assert len(fitted["keys"]) == len(fitted["labels"])
    for key, labels in zip(fitted["keys"], fitted["labels"]):
        assert len(labels) == LENGTHS[str(key)]


def test_held_out_sessions_are_listed_apart_from_the_keys(fitted):
    """Every session is either a key or held out, never both."""
    keys, held_out = {str(k) for k in fitted["keys"]}, {str(k) for k in fitted["hold_out_list"]}
    assert held_out and not keys & held_out
    assert keys | held_out == set(LENGTHS)

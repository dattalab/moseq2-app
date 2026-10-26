"""Cell 18: fit the AR-HMM.

    from moseq2_model.gui import learn_model_command

    learn_model_command(progress_paths)

Two lines, and everything the notebook has done so far feeds them. The model
reads the PC scores, fits an autoregressive hidden Markov model over them,
and labels every frame with the syllable it was in.

The sampler is stochastic and nothing here seeds it, so the labels differ run
to run and no value in the result is worth comparing against a stored one.
What is worth checking is everything the sampler is not free to change: how
many labels come back, which range they fall in, which parameters the fit
actually ran with, and whether the groups and uuids the earlier cells
established survived into the model. This follows what keypoint-moseq does
with its own sampler, where the model's structure and hyperparameters are
snapshotted and its states are only checked for being in range.

One detail that looks like corruption and is not: the first ``nlags`` frames
of every recording carry ``-5`` rather than a syllable. An autoregressive
model predicts a pose from the few before it, so the opening frames have
nothing to be predicted from and are left unlabelled.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from tests.conftest import MODEL_SETTINGS
from tests.extract_modeling.test_05_arena_mask import RECORDINGS
from tests.extract_modeling.test_09_aggregate import entries
from tests.extract_modeling.test_15_scores import uuids

FRAMES = 600

#: What an unlabelled frame carries instead of a syllable.
UNLABELLED = -5

#: The model's own record of how it was fit. Which hyperparameters the
#: sampler records alongside these varies, so they are checked for being
#: present rather than for being the whole set.
PARAMETERS = (
    "alpha",
    "ar_mat",
    "gamma",
    "kappa",
    "max_states",
    "model_class",
    "nlags",
    "nu",
)


@pytest.fixture(scope="session")
def fitted(model):
    """The model object, loaded once."""
    return joblib.load(model)


def test_the_model_is_written_where_the_cell_said(model, project):
    """1. The cell's output."""
    assert model == project / "models" / "model.p"
    assert model.stat().st_size > 0


def test_every_frame_of_every_recording_is_labelled(fitted, grouped):
    """2. The result: a syllable per frame, per recording.

    The labels are a list parallel to ``keys``, which is how they are matched
    back to recordings. Losing that alignment would misattribute every
    syllable in the analysis notebook without anything looking wrong.
    """
    assert sorted(fitted["keys"]) == sorted(uuids(grouped).values())
    assert len(fitted["labels"]) == len(RECORDINGS)

    for uuid, labels in zip(fitted["keys"], fitted["labels"]):
        labels = np.asarray(labels)
        assert labels.shape == (FRAMES,), uuid
        assert labels.dtype.kind == "i", uuid


def test_the_syllables_are_within_the_cap(fitted):
    """3. The model may not invent a syllable it was not allowed.

    ``max_states`` is what cell 17 set. A label at or past it would mean the
    labels and the transition matrix disagree about how many syllables exist.
    """
    labelled = np.concatenate(
        [np.asarray(labels) for labels in fitted["labels"]]
    )
    used = labelled[labelled != UNLABELLED]

    assert used.min() >= 0
    assert used.max() < MODEL_SETTINGS["max_states"]
    assert len(np.unique(used)) > 1, "the model put everything in one syllable"


def test_the_opening_frames_are_left_unlabelled(fitted):
    """4. What looks like corruption and is not.

    An autoregressive model predicts a pose from the ``nlags`` before it, so
    the first ``nlags`` frames of a recording have nothing to be predicted
    from. They carry a sentinel rather than a syllable, at the start of every
    recording and nowhere else.
    """
    nlags = fitted["model_parameters"]["nlags"]

    for uuid, labels in zip(fitted["keys"], fitted["labels"]):
        labels = np.asarray(labels)
        unlabelled = np.flatnonzero(labels == UNLABELLED)
        np.testing.assert_array_equal(unlabelled, np.arange(nlags), err_msg=uuid)
        assert (labels[nlags:] >= 0).all(), uuid


def test_the_fit_used_the_parameters_cell_17_set(fitted):
    """5. The contract between cell 17 and this one.

    The model records what it ran with, which is the only way to tell a
    configuration that was read from one that was silently defaulted.
    """
    parameters = fitted["model_parameters"]

    assert set(PARAMETERS) <= set(parameters), sorted(set(PARAMETERS) - set(parameters))
    assert parameters["max_states"] == MODEL_SETTINGS["max_states"]
    assert parameters["model_class"] == "CorrectedRobustARHMM"
    assert parameters["nlags"] > 0


def test_leaving_kappa_unset_means_the_length_of_the_data(fitted):
    """6. What ``kappa = None`` resolves to.

    The notebook says kappa defaults to the number of frames in the dataset.
    Four recordings of six hundred frames is twenty-four hundred, and that is
    what the model was fit with.
    """
    assert fitted["model_parameters"]["kappa"] == FRAMES * len(RECORDINGS)


def test_asking_for_a_robust_model_gets_one(fitted):
    """7. ``robust=True`` is the Student's t sampler, not a label.

    The class name is how the rest of the project tells which sampler ran,
    and the degrees-of-freedom parameter only exists on the robust one.
    """
    assert MODEL_SETTINGS["robust"] is True
    assert type(fitted["model"]).__name__ == "CorrectedRobustARHMM"
    assert fitted["model_parameters"]["nu"] is not None


def test_the_model_works_in_the_number_of_components_it_was_given(fitted):
    """8. ``npcs`` reaches the sampler.

    The whitening the model applies is sized by the number of components, so
    it is where a mismatch between cell 17 and the fit would show.
    """
    npcs = MODEL_SETTINGS["npcs"]
    whitening = fitted["whitening_parameters"]

    assert np.asarray(whitening["mu"]).shape == (npcs,)
    assert np.asarray(whitening["L"]).shape == (npcs, npcs)
    assert np.asarray(fitted["model_parameters"]["ar_mat"][0]).shape[0] == npcs


def test_the_groups_from_the_index_reach_the_model(fitted, grouped):
    """9. What cell 10 did has to survive this far.

    The analysis notebook compares syllable usage between conditions by
    reading the groups off the model. If they arrived as 'default' the
    comparison would run and return nothing.
    """
    expected = {
        entry["uuid"]: entry["group"] for entry in entries(grouped).values()
    }
    assert fitted["metadata"]["groups"] == expected
    assert set(expected.values()) == {"saline", "amph"}


def test_the_fit_reports_a_likelihood_per_recording(fitted):
    """10. The numbers a user watches to see whether the fit converged."""
    assert len(fitted["train_ll"]) == len(RECORDINGS)
    assert np.isfinite(fitted["train_ll"]).all()
    assert np.isfinite(fitted["loglikes"])


def test_nothing_was_held_out(fitted):
    """11. ``hold_out=False``, so every recording trained the model."""
    assert MODEL_SETTINGS["hold_out"] is False
    assert fitted["hold_out_list"] == []
    assert fitted["heldout_ll"] == []
    assert sorted(fitted["train_list"]) == sorted(fitted["keys"])

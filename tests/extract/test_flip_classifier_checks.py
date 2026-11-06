"""Extraction checks its flip classifier before any work.

A flip classifier is trained on frames of one crop size, cleaned one way. A
classifier trained in the Flip-Classifier-Training notebook
(``moseq2_app.steps.train_flip_classifier``) carries both: the crop size and
the cleaning settings each labeled session was cleaned with. Extraction stops
when its crop size is not the classifier's, naming both values. A session whose
cleaning settings equal none of the sets the classifier was trained with gets
one warning naming the session, each setting that differs, its value in each
trained set and its value now, and is extracted with the classifier. The
published classifiers that ``download-flip-file`` fetches carry only their
feature count, which the crop size check reads, and no cleaning settings, so
their cleaning is not checked.

A ``flip_classifier`` that does not exist, or cannot be read as a classifier,
stops extraction too: 1.5.0 skipped the check and extracted without flips
being corrected, or failed later.
"""

from __future__ import annotations

import pickle
import shutil
import types
import warnings
from dataclasses import replace

import joblib

import h5py
import numpy as np
import pytest

from moseq2_extract.helpers.wrappers import extract_wrapper, validate_flip_classifier
from moseq2_extract.options import ExtractOptions


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A classifier trained on 20 x 20 frames, cleaned with extraction's default settings."""
    from moseq2_app.steps import train_flip_classifier

    root = tmp_path_factory.mktemp("flip")
    rng = np.random.default_rng(0)
    session = root / "session.h5"
    with h5py.File(session, "w") as f:
        frames = np.zeros((30, 20, 20), dtype=np.uint8)
        frames[:, 8:12, 2:10] = 50
        frames[:, 6:14, 10:16] = 80
        f["frames"] = frames + rng.integers(0, 5, size=frames.shape, dtype=np.uint8)
    index = root / "flip-training-frame-ranges.p"
    with index.open("wb") as f:
        pickle.dump(({"session": str(session)}, {"session": [(False, range(0, 30))]}), f)
    return str(train_flip_classifier(str(index), str(root / "flip.p")).classifier)


def settings(**changes):
    return {**replace(ExtractOptions(), **changes).as_dict(), "bg_roi_index": 0}


def extract(tmp_path, config):
    """Extraction of a recording that is never reached: the check comes first."""
    return extract_wrapper(str(tmp_path / "session" / "depth.avi"), "proc", config)


def test_a_missing_flip_classifier_is_an_error(tmp_path):
    missing = tmp_path / "flip.p"

    with pytest.raises(FileNotFoundError) as e:
        extract(tmp_path, settings(flip_classifier=str(missing)))

    assert str(missing) in str(e.value)
    assert "download-flip-file" in str(e.value)


def test_a_flip_classifier_that_cannot_be_read_is_an_error(tmp_path):
    broken = tmp_path / "flip.p"
    broken.write_bytes(b"not a classifier")

    with pytest.raises(ValueError) as e:
        extract(tmp_path, settings(flip_classifier=str(broken)))

    assert str(broken) in str(e.value)


def test_extraction_stops_when_the_flip_classifier_crop_size_differs(tmp_path, trained):
    with pytest.raises(ValueError) as e:
        extract(tmp_path, settings(flip_classifier=trained, crop_size=(80, 80)))

    message = str(e.value)
    assert trained in message
    assert "(80, 80)" in message and "(20, 20)" in message


def cleaning_warnings(caught):
    from moseq2_extract.helpers.wrappers import FlipClassifierCleaningWarning

    return [str(w.message) for w in caught if issubclass(w.category, FlipClassifierCleaningWarning)]


def test_extraction_with_other_cleaning_warns_once_and_completes(tmp_path, raw_sessions, trained):
    session = tmp_path / "saline_example_0"
    shutil.copytree(f"{raw_sessions}/saline_example_0", session)
    recording = str(session / "depth.avi")
    config = settings(flip_classifier=trained, crop_size=(20, 20), spatial_filter_size=(7,), tail_filter_iters=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = extract_wrapper(recording, "proc", config, num_frames=40)

    [message] = cleaning_warnings(caught)
    assert trained in message and recording in message
    assert "spatial_filter_size is [7] now, [3] in training" in message
    assert "tail_filter_iters is 2 now, 1 in training" in message
    assert "tail_filter_size" not in message
    assert "config.yaml" in message and "session_config.yaml" in message
    assert "train the flip classifier again" in message
    with h5py.File(f"{written}/results.h5", "r") as f:
        assert f["frames"].shape == (40, 20, 20)


def test_a_session_matching_any_trained_set_passes_and_one_matching_none_names_each(tmp_path, trained):
    """A classifier trained on sessions cleaned two ways."""
    from moseq2_common.frames import CleaningOptions

    two_ways = tmp_path / "flip.p"
    clf = joblib.load(trained)
    default = CleaningOptions().as_dict()
    clf.training_settings = {**clf.training_settings,
                             "cleaning": [default, {**default, "spatial_filter_size": [5]}]}
    joblib.dump(clf, two_ways)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_flip_classifier(settings(flip_classifier=str(two_ways), crop_size=(20, 20),
                                          spatial_filter_size=(5,)), "cohort1/depth.avi")
        assert cleaning_warnings(caught) == []

        validate_flip_classifier(settings(flip_classifier=str(two_ways), crop_size=(20, 20),
                                          spatial_filter_size=(7,)), "cohort1/depth.avi")
    [message] = cleaning_warnings(caught)
    assert "cohort1/depth.avi" in message
    assert "cleaned in 2 ways" in message
    assert "spatial_filter_size is [7] now, [3] and [5] in training" in message


def test_a_classifier_without_cleaning_settings_is_not_cleaning_checked(tmp_path):
    """A downloaded classifier, or one trained before classifiers carried their settings: crop size only."""
    joblib.dump(types.SimpleNamespace(training_settings={"crop_size": [80, 80]}), tmp_path / "flip.p")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_flip_classifier(settings(flip_classifier=str(tmp_path / "flip.p"), crop_size=(80, 80),
                                          spatial_filter_size=(7,), tail_filter_iters=4))
    assert cleaning_warnings(caught) == []

    with pytest.raises(ValueError, match=r"cropped to \(80, 80\), but crop_size is \(20, 20\)"):
        validate_flip_classifier(settings(flip_classifier=str(tmp_path / "flip.p"), crop_size=(20, 20)))


def test_a_classifier_trained_with_these_settings_passes(trained):
    validate_flip_classifier(settings(flip_classifier=trained, crop_size=(20, 20)))

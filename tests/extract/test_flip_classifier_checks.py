"""Extraction checks its flip classifier before any work.

A flip classifier is trained on frames of one crop size, cleaned one way. A
classifier trained in the Flip-Classifier-Training notebook
(``moseq2_app.steps.train_flip_classifier``) carries both: the crop size and
the cleaning settings each labeled session was cleaned with. Extraction stops
when its crop size, or its cleaning settings, are not ones the classifier was
trained with, naming both values. The published classifiers that
``download-flip-file`` fetches carry only their feature count, which the crop
size check reads, and no cleaning settings.

A ``flip_classifier`` that does not exist, or cannot be read as a classifier,
stops extraction too: 1.5.0 skipped the check and extracted without flips
being corrected, or failed later.
"""

from __future__ import annotations

import pickle
from dataclasses import replace

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


def test_extraction_stops_when_the_flip_classifier_was_trained_with_other_cleaning(tmp_path, trained):
    with pytest.raises(ValueError) as e:
        extract(tmp_path, settings(flip_classifier=trained, crop_size=(20, 20), spatial_filter_size=(7,)))

    message = str(e.value)
    assert trained in message
    assert "spatial_filter_size" in message and "[7]" in message and "[3]" in message


def test_a_classifier_trained_with_these_settings_passes(trained):
    validate_flip_classifier(settings(flip_classifier=trained, crop_size=(20, 20)))

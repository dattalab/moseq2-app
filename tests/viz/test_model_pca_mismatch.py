"""A model whose labels do not line up with the PCA scores the index names.

The analysis notebook's first table (``scalars_to_dataframe`` with a model)
joins each session's syllable labels to the frame index in the PCA scores
file the index's ``pca_path`` names. That only works when the model was fit
on those scores: every session the model labels has to be in the file, with
one score row per label. When PCA is run again after the model was fit, or
the index points at another run's scores, the join is refused, and the error
has to say which sessions failed, how, and what to do about it.
"""

from __future__ import annotations

import h5py
import joblib
import numpy as np
import pytest

from moseq2_viz.model.util import prepare_model_dataframe
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.util import parse_index

UUIDS = ["uuid-a", "uuid-b", "uuid-c"]
LENGTH = 50


def write_model(path, uuids=UUIDS, length=LENGTH, pc_score_path=None):
    labels = [np.arange(length) % 3 for _ in uuids]
    model = {
        "labels": labels,
        "train_list": list(uuids),
        "keys": list(uuids),
        "metadata": {"uuids": list(uuids), "groups": {u: "default" for u in uuids}},
        "model_parameters": {"kappa": 100, "model_class": "ARHMM"},
    }
    if pc_score_path is not None:
        model["pc_score_path"] = str(pc_score_path)
    joblib.dump(model, path)
    return path


def write_scores(path, lengths):
    with h5py.File(path, "w") as f:
        for uuid, length in lengths.items():
            f[f"scores_idx/{uuid}"] = np.arange(length, dtype="float32")
    return path


def test_matching_scores_give_one_row_per_label(tmp_path):
    """The control: a model and the scores it was fit on."""
    model = write_model(tmp_path / "model.p")
    scores = write_scores(tmp_path / "pca_scores.h5", {u: LENGTH for u in UUIDS})

    df = prepare_model_dataframe(str(model), str(scores))
    assert len(df) == len(UUIDS) * LENGTH


def test_every_session_that_does_not_line_up_is_named(tmp_path):
    """A missing UUID and a length mismatch, reported together with the file."""
    fitted_on = tmp_path / "old_pca" / "pca_scores.h5"
    model = write_model(tmp_path / "model.p", pc_score_path=fitted_on)
    scores = write_scores(tmp_path / "pca_scores.h5", {"uuid-a": LENGTH, "uuid-c": LENGTH - 2})

    with pytest.raises(ValueError) as raised:
        prepare_model_dataframe(str(model), str(scores))
    message = str(raised.value)

    assert str(scores) in message
    assert "2 of the 3 sessions" in message
    assert "uuid-b: its UUID is not in the PCA scores file" in message
    assert f"uuid-c: {LENGTH} labels in the model, {LENGTH - 2} frames in the PCA scores file" in message
    assert "uuid-a" not in message
    assert str(fitted_on) in message
    assert "PCA was run again after the model was fit" in message
    assert "pca_path" in message
    assert "fit the model again" in message


def test_sessions_are_named_from_the_index(analysis_bundle, tmp_path):
    """Through the analysis notebook's table, with the twenty-session project.

    The index carries each session's name, so the message gives it beside
    the UUID. The scores file here is the project's own with one session
    removed and another two frames short, as a different PCA run would be.
    """
    _, index = parse_index(str(analysis_bundle / "moseq2-index.yaml"))
    uuids = sorted(index["files"])
    with h5py.File(index["pca_path"], "r") as f:
        lengths = {u: len(f[f"scores_idx/{u}"]) for u in uuids}
    missing, short = uuids[0], uuids[1]
    del lengths[missing]
    lengths[short] -= 2
    index = {**index, "pca_path": str(write_scores(tmp_path / "pca_scores.h5", lengths))}

    with pytest.raises(ValueError) as raised:
        scalars_to_dataframe(index, model_path=str(analysis_bundle / "models" / "model.p"))
    message = str(raised.value)

    for uuid in (missing, short):
        name = index["files"][uuid]["metadata"]["SessionName"]
        assert f"{uuid} ({name})" in message
    assert f"2 of the {len(uuids)} sessions" in message
    assert index["pca_path"] in message

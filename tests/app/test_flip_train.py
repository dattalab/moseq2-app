"""Building the flip classifier's training set.

The Flip-Classifier-Training notebook hands the frame ranges a user labeled
to ``moseq2_app.steps.train_flip_classifier``, whose ``create_training_dataset``
assembles them into one array, reports its shape and size, and saves it for
training. The suite walks neither that
notebook nor this module, so the function is tested here directly.
"""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest

from moseq2_app.flip import train
from moseq2_common.frames import CleaningOptions
from moseq2_common.yaml import write_yaml


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

    train.create_training_dataset(str(index), CleaningOptions())

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


def test_the_cleaned_third_follows_the_cleaning_parameters(tmp_path):
    """Training cleans with the parameters it is given, cable erosion included.

    Extraction erodes the mouse ``cable_filter_iters`` times before the
    classifier sees it, so a classifier trained on uneroded frames is applied
    to frames unlike its training set. With two erosions the cleaned third
    differs from the defaults' and is what ``clean_frames`` makes with the
    same settings.
    """
    import cv2

    from moseq2_extract.extract.proc import clean_frames

    rng = np.random.default_rng(0)
    labeled = np.zeros((10, 40, 40), dtype=np.uint8)
    labeled[:, 12:28, 8:32] = 60
    labeled += rng.integers(0, 20, size=labeled.shape, dtype=np.uint8)
    session, index = tmp_path / "session.h5", tmp_path / "flip-training-frame-ranges.p"
    with h5py.File(session, "w") as f:
        f["frames"] = labeled
    with index.open("wb") as f:
        pickle.dump(({"session": str(session)}, {"session": [(False, range(0, 10))]}), f)

    def cleaned_third(cleaning):
        with np.load(train.create_training_dataset(str(index), cleaning)) as data:
            frames = data["frames"]
        return frames[2 * len(frames) // 3 :]

    eroded = CleaningOptions(cable_filter_iters=2)
    both_ways = np.concatenate((labeled, np.rot90(labeled, 2, axes=(1, 2))), axis=0)
    expected = clean_frames(
        np.where(both_ways < eroded.min_height, 0, both_ways),
        (3,),
        (0,),
        strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iters_tail=1,
        strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iters_min=2,
    )

    np.testing.assert_array_equal(cleaned_third(eroded), expected)
    assert not np.array_equal(cleaned_third(eroded), cleaned_third(CleaningOptions()))


# ------------------------------------------------------------------ each session's own cleaning settings


def _extracted_session(root, name, uuid, frames, stored):
    """A recording in ``root/name`` and its extraction in its session folder in ``aggregate_results/``:
    ``results.h5`` holding ``frames``, the uuid and the cleaning settings ``stored`` as extraction
    stores its settings, and ``results.yaml`` beside it."""
    folder = root / "aggregate_results" / name
    folder.mkdir(parents=True)
    (root / name).mkdir()
    (root / name / "depth.dat").touch()
    h5 = folder / "results.h5"
    with h5py.File(h5, "w") as f:
        f["frames"] = frames
        f["metadata/uuid"] = uuid
        for key, value in stored.items():
            f[f"metadata/extraction/parameters/{key}"] = value
    write_yaml({"complete": True, "skip": False, "uuid": uuid}, str(folder / "results.yaml"))
    return h5


#: Cleaning settings as extraction stores them for an extraction run with the defaults.
DEFAULT_STORED = {"spatial_filter_size": [3], "temporal_filter_size": [0], "tail_filter_shape": "ellipse",
                  "tail_filter_size": [9, 9], "tail_filter_iters": 1, "cable_filter_size": [5, 5],
                  "cable_filter_shape": "rectangle", "cable_filter_iters": 0, "min_height": 10}


@pytest.fixture
def flip_project(tmp_path):
    """Two extracted sessions, each with five labeled frames, listed in moseq2-project.yaml, and a
    config.yaml; session_b's arena-tool entry in session_config.yaml sets a spatial filter of 5."""
    from moseq2_extract.gui import generate_config_command

    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    rng = np.random.default_rng(0)
    h5s = {}
    for i, name in enumerate(("session_a", "session_b")):
        frames = rng.integers(0, 80, size=(10, 20, 20)).astype(np.uint8)
        h5s[name] = _extracted_session(root, name, f"uuid-{i}", frames, DEFAULT_STORED)
    write_yaml({"sessions": {f"{name}/depth.dat": {"recording": f"{name}/depth.dat",
                                                   "folder": h5.parent.relative_to(root).as_posix(),
                                                   "uuid": f"uuid-{i}"}
                             for i, (name, h5) in enumerate(h5s.items())}},
               str(root / "moseq2-project.yaml"))
    write_yaml({"session_b/depth.dat": {"spatial_filter_size": [5]}}, str(root / "session_config.yaml"))
    index = root / "flip-training-frame-ranges.p"
    with index.open("wb") as f:
        pickle.dump(({n: str(p) for n, p in h5s.items()}, {n: [(False, range(0, 5))] for n in h5s}), f)
    return root, index


def _cleaning_calls(monkeypatch):
    """The spatial filter each call to ``clean_frames`` cleans with, and how many frames it cleans."""
    calls = []

    def record(frames, prefilter_space, *args, **kwargs):
        calls.append((tuple(prefilter_space), len(frames)))
        return frames

    monkeypatch.setattr(train, "clean_frames", record)
    return calls


def test_training_cleans_each_session_with_its_own_settings(flip_project, monkeypatch, tmp_path):
    """session_a takes config.yaml's spatial filter, 3; session_b its entry's, 5: each session's
    frames, both ways round, cleaned as that recording is extracted."""
    from moseq2_app.steps import train_flip_classifier

    root, index = flip_project
    calls = _cleaning_calls(monkeypatch)
    monkeypatch.setattr(train, "train_classifier", lambda *a, **kw: None)
    monkeypatch.setattr(train, "save_classifier", lambda *a, **kw: None)

    train_flip_classifier(str(index), str(tmp_path / "flip.p"), str(root / "config.yaml"))

    assert sorted(calls) == [((3,), 10), ((5,), 10)]


def test_training_warns_when_a_session_was_extracted_with_other_cleaning_settings(flip_project, monkeypatch,
                                                                                    tmp_path, capsys):
    """session_b's extraction ran with a spatial filter of 3, and its settings now resolve to 5:
    training says so, naming the session, the setting and both values, suggests re-extracting,
    and cleans with 5. session_a, extracted with what it resolves to, is not named."""
    from moseq2_app.steps import train_flip_classifier

    root, index = flip_project
    calls = _cleaning_calls(monkeypatch)
    monkeypatch.setattr(train, "train_classifier", lambda *a, **kw: None)
    monkeypatch.setattr(train, "save_classifier", lambda *a, **kw: None)

    train_flip_classifier(str(index), str(tmp_path / "flip.p"), str(root / "config.yaml"))

    out = capsys.readouterr().out
    assert "session_b" in out
    assert "spatial_filter_size" in out
    assert "[3]" in out and "[5]" in out
    assert "re-extract" in out
    assert "session_a" not in out
    assert ((5,), 10) in calls


def test_a_sessions_recording_is_found_through_the_bookkeeping_file(tmp_path, monkeypatch):
    """Labeled sessions are extraction files with no uuid and no results yaml beside them: the project's
    moseq2-project.yaml says which recording each one is, so session_b's arena-tool entry reaches training."""
    from moseq2_app.steps import train_flip_classifier
    from moseq2_extract.gui import generate_config_command

    root = tmp_path / "project"
    root.mkdir(parents=True)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    rng = np.random.default_rng(0)
    h5s, sessions = {}, {}
    for i, name in enumerate(("session_a", "session_b")):
        (root / name).mkdir()
        (root / name / "depth.dat").touch()
        h5 = root / "aggregate_results" / name / "results.h5"
        h5.parent.mkdir(parents=True)
        with h5py.File(h5, "w") as f:
            f["frames"] = rng.integers(0, 80, size=(10, 20, 20)).astype(np.uint8)
        h5s[name] = h5
        key = f"{name}/depth.dat"
        sessions[key] = {"recording": key, "folder": f"aggregate_results/{name}", "uuid": f"uuid-{i}"}
    write_yaml({"sessions": sessions}, str(root / "moseq2-project.yaml"))
    write_yaml({"session_b/depth.dat": {"spatial_filter_size": [5]}}, str(root / "session_config.yaml"))
    index = root / "flip-training-frame-ranges.p"
    with index.open("wb") as f:
        pickle.dump(({n: str(p) for n, p in h5s.items()}, {n: [(False, range(0, 5))] for n in h5s}), f)

    calls = _cleaning_calls(monkeypatch)
    monkeypatch.setattr(train, "train_classifier", lambda *a, **kw: None)
    monkeypatch.setattr(train, "save_classifier", lambda *a, **kw: None)

    train_flip_classifier(str(index), str(tmp_path / "flip.p"), str(root / "config.yaml"))

    assert sorted(calls) == [((3,), 10), ((5,), 10)]


def test_a_labeled_session_the_project_does_not_list_is_named(flip_project, monkeypatch, tmp_path, capsys):
    """A labeled file that is none of the project's sessions has no recording, so no arena-tool entry:
    training says so, naming the file and the bookkeeping file, and cleans it with config.yaml's settings."""
    from moseq2_app.steps import train_flip_classifier

    root, index = flip_project
    (root / "moseq2-project.yaml").write_text("sessions: {}\n")
    calls = _cleaning_calls(monkeypatch)
    monkeypatch.setattr(train, "train_classifier", lambda *a, **kw: None)
    monkeypatch.setattr(train, "save_classifier", lambda *a, **kw: None)

    train_flip_classifier(str(index), str(tmp_path / "flip.p"), str(root / "config.yaml"))

    out = capsys.readouterr().out
    assert str(root / "aggregate_results" / "session_b" / "results.h5") in out
    assert str(root / "moseq2-project.yaml") in out
    assert {spatial for spatial, _ in calls} == {(3,)}
    assert sum(frames for _, frames in calls) == 20

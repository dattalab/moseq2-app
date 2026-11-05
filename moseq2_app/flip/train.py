import os
import h5py
import pickle
import joblib
import numpy as np
from pathlib import Path
from collections.abc import Mapping
from dataclasses import fields, replace
from functools import partial
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from moseq2_common.bookkeeping import load, resolve_path, session_h5
from moseq2_common.frames import CleaningOptions, cleaning_arguments, odd_kernel
from moseq2_common.h5 import h5_to_dict
from moseq2_common.options import read_settings
from moseq2_common.project import find_project
from moseq2_extract.extract.proc import clean_frames
from moseq2_extract.options import EXTRACT_STAGE
from moseq2_extract.session_config import read_entries, recording_entry
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from moseq2_common.project import outdated_names

#: 1.5's ``CleanParameters``, which the 1.5 flip notebook imports, stops with a message to download
#: the 1.6 notebooks (its settings are extraction's, ``moseq2_common.frames.CleaningOptions``).
__getattr__ = outdated_names(__name__, ("CleanParameters",))


#: How a user gets labels from the widget into training, as every message about
#: unsaved labels says it.
SAVE_AND_RETRAIN = (
    "click Save Labels (or run the notebook's save cell), then run training again"
)


def _load_frame_ranges(data_index_path: Path):
    """The saved frame ranges: ({session: .h5 path}, {session: [(left, frames), ...]})."""
    if not data_index_path.exists():
        raise FileNotFoundError(
            f"No saved flip labels at {data_index_path}: the file does not exist. "
            "Labels reach training only once they are saved: in the labeling widget, "
            f"{SAVE_AND_RETRAIN}."
        )
    with open(data_index_path, "rb") as f:
        return pickle.load(f)


def _plain(value):
    """``value`` as YAML would hold it, so a stored array and a resolved tuple compare equal."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _show(value):
    """A value written as it appears in config.yaml."""
    if isinstance(value, list):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    return str(value)


def _recordings_by_file(project):
    """Each session's recording in ``project``, by its extraction file, as ``moseq2-project.yaml`` lists them."""
    return {os.path.normcase(str(session_h5(project, session))): str(resolve_path(project, session.recording))
            for session in load(project).sessions.values()}


def _stored_cleaning(h5):
    """The cleaning settings extraction stored in the session's ``.h5``, or None when it stored none."""
    with h5py.File(h5, "r") as f:
        if "metadata/extraction/parameters" not in f:
            return None
        stored = h5_to_dict(f, "metadata/extraction/parameters")
    return {f.name: stored[f.name] for f in fields(CleaningOptions) if f.name in stored}


def _check_extraction(name, h5, cleaning, sources, recording):
    """Say so when ``name``'s extraction ran with cleaning settings other than ``cleaning``."""
    stored = _stored_cleaning(h5)
    if not stored:
        return
    differ = [(setting, _plain(value), _plain(getattr(cleaning, setting))) for setting, value in stored.items()
              if _plain(value) != _plain(getattr(cleaning, setting))]
    if not differ:
        return
    listed = "; ".join(f"{setting} {_show(then)} at extraction, {_show(now)} now" for setting, then, now in differ)
    print(f"{name} ({h5}) was extracted with cleaning settings other than the ones it has now, from {sources}: "
          f"{listed}. Training cleans its frames with the settings it has now. If those are the settings you "
          f"extract with, re-extract {recording or name} so its frames are cleaned as the classifier will see them.")


def session_cleaning(data_index_path: str, config_file: str | None) -> dict:
    """Each labeled session's cleaning settings, resolved as extraction resolves them.

    config.yaml's ``extract`` section, with the arena-tool settings of the session's
    recording in the project's session_config.yaml over it; extraction's defaults
    without ``config_file``. A labeled file's recording is the one the project's
    ``moseq2-project.yaml`` lists for it. Where a session's extraction stored other cleaning settings than
    the ones it resolves to, a message names each, with both values, and suggests
    re-extracting; the resolved settings are the ones returned.

    Args:
    data_index_path (str): the saved frame ranges (``flip-training-frame-ranges.p``).
    config_file (str): the project's config.yaml, or None.

    Returns:
    (dict): {session: CleaningOptions}, for each session with labeled frames.
    """
    session_paths, data_index = _load_frame_ranges(Path(data_index_path))
    project = find_project(config_file) if config_file else None
    config = read_settings(config_file, CleaningOptions, stage_classes=EXTRACT_STAGE) if config_file else {}
    entries = read_entries(project) if project is not None else {}
    recordings = _recordings_by_file(project) if entries else {}
    if config_file:
        sources = f"{config_file}'s extract section and its recording's session_config.yaml entry"
    else:
        sources = "extraction's defaults"

    cleaning = {}
    for name in data_index:
        h5 = session_paths[name]
        recording = recordings.get(os.path.normcase(os.path.abspath(h5))) if entries else None
        if entries and recording is None:
            print(f"{name} ({h5}) is not one of the sessions {project.bookkeeping_file} lists, so its recording "
                  f"is not known and its frames are cleaned with {config_file}'s extract section alone, without "
                  "a session_config.yaml entry. To clean them as their recording is extracted, label the "
                  "sessions in the project's aggregate_results folder.")
        entry = recording_entry(project, recording, entries) if recording is not None else {}
        cleaning[name] = CleaningOptions.resolve(config=config, entry=entry)
        _check_extraction(name, h5, cleaning[name], sources, recording)
    return cleaning


def create_training_dataset(data_index_path: str, cleaning) -> Path:
    """Assemble the labeled frames into a training set and save it beside ``data_index_path``.

    The set holds each labeled frame and its 180 degree rotation, a randomly
    shifted copy of both, and a copy of both cleaned as extraction cleans its
    frames, each session's with its own settings in ``cleaning``.

    Args:
    data_index_path (str): the saved frame ranges (``flip-training-frame-ranges.p``).
    cleaning (Mapping or CleaningOptions): {session: CleaningOptions} (``session_cleaning``),
        or one CleaningOptions for every session.

    Returns:
    out_path (Path): the training set, ``training_data.npz``.
    """
    np.random.seed(0)
    data_index_path = Path(data_index_path)
    out_path = data_index_path.with_name("training_data.npz")

    # load trainingdata index
    session_paths, data_index = _load_frame_ranges(data_index_path)
    if not isinstance(cleaning, Mapping):
        cleaning = {k: cleaning for k in data_index}

    # load frames, and the session each came from
    frames = []
    sessions = []
    for k, v in data_index.items():
        with h5py.File(session_paths[k], "r") as h5f:
            for left, _slice in v:
                frames_subset = h5f["frames"][_slice]
                if left:
                    frames_subset = np.rot90(frames_subset, 2, axes=(1, 2))
                frames.append(frames_subset)
                sessions += [k] * len(frames_subset)
    if sum(len(f) for f in frames) == 0:
        raise ValueError(
            f"No labeled frames in {data_index_path}. Labels reach training only once "
            "they are saved: if the labeling widget lists ranges under Selected Ranges, "
            f"{SAVE_AND_RETRAIN}. If it lists none, add a frame range first: click "
            "Start Range, step to the last frame of the range, then click Facing Left "
            "or Facing Right, and save."
        )
    frames = np.concatenate(frames, axis=0)
    # rotate frames
    frames = np.concatenate((frames, np.rot90(frames, 2, axes=(1, 2))), axis=0)
    sessions = sessions + sessions

    flipped = np.zeros((len(frames),), dtype=np.uint8)
    flipped[len(frames) // 2 :] = 1

    # add some randomly shifted frames
    shifts = np.random.randint(-5, 5, size=(len(frames), 2))
    shifted_frames = np.array([np.roll(f, tuple(s), axis=(0, 1)) for f, s in zip(frames, shifts)]).astype(np.uint8)

    # remove noise from frames as extraction does: heights below min_height to zero, then clean_frames,
    # with an even median kernel rounded up to the next odd size. Sessions with the same settings
    # are cleaned together, in the order their frames are in.
    cleaned_frames = np.empty_like(frames)
    for options in dict.fromkeys(cleaning[k] for k in data_index):
        group = np.array([cleaning[k] == options for k in sessions])
        options = replace(
            options,
            spatial_filter_size=tuple(odd_kernel(k) for k in options.spatial_filter_size),
            temporal_filter_size=tuple(odd_kernel(k) for k in options.temporal_filter_size),
        )
        cleaned_frames[group] = clean_frames(
            np.where(frames[group] < options.min_height, 0, frames[group]),
            **cleaning_arguments(options),
            frame_dtype=frames.dtype,
        )
    frames = np.concatenate((frames, shifted_frames, cleaned_frames), axis=0)
    print(
        f"Training data shape: {frames.shape}; memory usage: {frames.nbytes / 1e9:0.2f} GB"
    )

    flipped = np.concatenate((flipped, flipped, flipped), axis=0)

    np.savez(out_path, frames=frames, flipped=flipped)
    return out_path


def flatten(array: np.ndarray) -> np.ndarray:
    return array.reshape(len(array), -1)


def batch_apply_pca(frames: np.ndarray, pca: PCA, batch_size: int = 1000) -> np.ndarray:
    output = []
    if len(frames) < batch_size:
        return pca.transform(flatten(frames)).astype(np.float32)

    for arr in np.array_split(frames, len(frames) // batch_size):
        output.append(pca.transform(flatten(arr)).astype(np.float32))
    return np.concatenate(output, axis=0).astype(np.float32)


def train_classifier(
    data_path: str,
    classifier: str = "SVM",
    n_components: int = 20,
):
    """Train a classifier to predict the orientation of a mouse.
    Parameters:
        data_path (str): Path to the training data numpy file.
        classifier (str): Classifier to use. Either 'SVM' or 'RF'.
        n_components (int): Number of components to keep in PCA."""
    data = np.load(data_path)
    frames = data["frames"]
    flipped = data["flipped"]

    print("Fitting PCA")
    pca = PCA(n_components=n_components)
    pca.fit(flatten(frames[-len(frames) // 3:]))

    # The fitted PCA is bound into the function rather than passed as kw_args:
    # cross-validation clones the pipeline per fold, and scikit-learn's clone
    # replaces an estimator inside a kw_args dict with an unfitted copy.
    pipeline = make_pipeline(
        FunctionTransformer(partial(batch_apply_pca, pca=pca), validate=False),
        StandardScaler(),
        (
            RandomForestClassifier(n_estimators=150)
            if classifier == "RF"
            else SVC(probability=True)
        ),
    )

    print("Running cross-validation")
    accuracy = cross_val_score(
        pipeline, frames, flipped, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    print(f"Held-out model accuracy: {accuracy.mean()}")

    print("Final fitting step")
    return pipeline.fit(frames, flipped)


def save_classifier(clf_pipeline, out_path: str):
    joblib.dump(clf_pipeline, out_path)
    print(f"Classifier saved to {out_path}")

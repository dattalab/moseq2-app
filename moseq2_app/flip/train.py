import h5py
import pickle
import joblib
import numpy as np
from pathlib import Path
from dataclasses import replace
from functools import partial
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from moseq2_common.frames import CleaningOptions, cleaning_arguments, odd_kernel
from moseq2_extract.extract.proc import clean_frames
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer


#: How a user gets labels from the widget into training, as every message about
#: unsaved labels says it.
SAVE_AND_RETRAIN = (
    "click Save Labels (or run the notebook's save cell), then run training again"
)


def create_training_dataset(data_index_path: str, cleaning: CleaningOptions) -> Path:
    """Assemble the labeled frames into a training set and save it beside ``data_index_path``.

    The set holds each labeled frame and its 180 degree rotation, a randomly
    shifted copy of both, and a copy of both cleaned as extraction cleans its
    frames with the settings ``cleaning``.

    Args:
    data_index_path (str): the saved frame ranges (``flip-training-frame-ranges.p``).
    cleaning (CleaningOptions): extraction's cleaning settings.

    Returns:
    out_path (Path): the training set, ``training_data.npz``.
    """
    np.random.seed(0)
    data_index_path = Path(data_index_path)
    out_path = data_index_path.with_name("training_data.npz")

    # load trainingdata index
    if not data_index_path.exists():
        raise FileNotFoundError(
            f"No saved flip labels at {data_index_path}: the file does not exist. "
            "Labels reach training only once they are saved: in the labeling widget, "
            f"{SAVE_AND_RETRAIN}."
        )
    with open(data_index_path, "rb") as f:
        session_paths, data_index = pickle.load(f)

    # load frames
    frames = []
    for k, v in data_index.items():
        with h5py.File(session_paths[k], "r") as h5f:
            for left, _slice in v:
                frames_subset = h5f["frames"][_slice]
                if left:
                    frames_subset = np.rot90(frames_subset, 2, axes=(1, 2))
                frames.append(frames_subset)
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

    flipped = np.zeros((len(frames),), dtype=np.uint8)
    flipped[len(frames) // 2 :] = 1

    # add some randomly shifted frames
    shifts = np.random.randint(-5, 5, size=(len(frames), 2))
    shifted_frames = np.array([np.roll(f, tuple(s), axis=(0, 1)) for f, s in zip(frames, shifts)]).astype(np.uint8)

    # remove noise from frames as extraction does: heights below min_height to zero, then clean_frames,
    # with an even median kernel rounded up to the next odd size
    cleaning = replace(
        cleaning,
        spatial_filter_size=tuple(odd_kernel(k) for k in cleaning.spatial_filter_size),
        temporal_filter_size=tuple(odd_kernel(k) for k in cleaning.temporal_filter_size),
    )
    cleaned_frames = clean_frames(
        np.where(frames < cleaning.min_height, 0, frames),
        **cleaning_arguments(cleaning),
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

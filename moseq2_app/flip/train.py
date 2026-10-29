import cv2
import h5py
import pickle
import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from moseq2_extract.extract.proc import clean_frames
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer


@dataclass
class CleanParameters:
    prefilter_space: tuple = (3,)
    prefilter_time: Optional[tuple] = None
    strel_tail: Tuple[int, int] = (9, 9)
    strel_min: Tuple[int, int] = (5, 5)
    iters_tail: Optional[int] = 1
    iters_min: Optional[int] = None
    height_threshold: int = 10


def create_training_dataset(
    data_index_path: str, clean_parameters: Optional[CleanParameters] = None
) -> str:
    np.random.seed(0)
    data_index_path = Path(data_index_path)
    out_path = data_index_path.with_name("training_data.npz")
    if clean_parameters is None:
        clean_parameters = CleanParameters()

    # load trainingdata index
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
    frames = np.concatenate(frames, axis=0)
    # rotate frames
    frames = np.concatenate((frames, np.rot90(frames, 2, axes=(1, 2))), axis=0)

    flipped = np.zeros((len(frames),), dtype=np.uint8)
    flipped[len(frames) // 2 :] = 1

    # add some randomly shifted frames
    shifts = np.random.randint(-5, 5, size=(len(frames), 2))
    shifted_frames = np.array([np.roll(f, tuple(s), axis=(0, 1)) for f, s in zip(frames, shifts)]).astype(np.uint8)

    # remove noise from frames
    cleaned_frames = clean_frames(
        np.where(frames > clean_parameters.height_threshold, frames, 0),
        clean_parameters.prefilter_space,
        clean_parameters.prefilter_time,
        strel_tail=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, clean_parameters.strel_tail
        ),
        iters_tail=clean_parameters.iters_tail,
        strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, clean_parameters.strel_min),
        iters_min=clean_parameters.iters_min,
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

    pipeline = make_pipeline(
        FunctionTransformer(batch_apply_pca, kw_args={"pca": pca}, validate=False),
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

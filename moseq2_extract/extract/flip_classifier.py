"""
Loading and evaluation of flip classifiers.

A flip classifier is a random forest over flattened cropped mouse frames. The
published classifiers were pickled by scikit-learn 0.19.0 and cannot be read by
a modern scikit-learn: the module layout moved, the Cython node struct grew a
field, and, past both of those, ``DecisionTreeClassifier.predict_proba`` stopped
normalising leaf values because modern trees store fractions at fit time while
these store raw weighted counts. A shim that got that far would return row sums
in the thousands rather than 1.0.

Classifiers are therefore carried as ``.npz`` archives of the forest's node
arrays, read with ``allow_pickle=False``, and evaluated by :func:`predict_proba`
below. That routine reproduces scikit-learn 0.20.3's serial evaluation term for
term, in stored tree order, and is bit-identical to it on real data.

``load_flip_classifier`` also accepts a pickle and hands it to ``joblib``
unchanged, so an environment with the original scikit-learn keeps the original
code path.

Ensembles are summed in tree order 0..n-1. scikit-learn dispatches its trees
through a thread pool and accumulates under a lock, so with the ``n_jobs=-1``
stored in the published classifiers its own sum order varies between calls and
its result varies in the last ulp. Tree order here is fixed, and matches what
scikit-learn produces when its pool is serial.
"""

import numpy as np

# scikit-learn casts to float32 before traversal (sklearn.tree.tree.DTYPE) and
# compares against float64 thresholds, promoting the operand. Both are kept.
_DTYPE = np.float32

_NPZ_MAGIC = b"PK\x03\x04"

_SUPPORTED_FORMAT_VERSIONS = (1,)

_REQUIRED_KEYS = (
    "format_version",
    "n_trees",
    "n_features",
    "n_classes",
    "n_outputs",
    "classes",
    "tree_offsets",
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "value",
)


class FlipClassifier:
    """A random forest flip classifier evaluated with numpy alone.

    Exposes the attributes ``moseq2_extract`` reads off a classifier:
    ``classes_``, ``n_features_``, ``n_classes_`` and ``predict_proba``.
    """

    def __init__(self, arrays, metadata=None):
        self.metadata = dict(metadata or {})

        self._n_trees = int(arrays["n_trees"])
        self.n_features_ = int(arrays["n_features"])
        self.n_classes_ = int(arrays["n_classes"])
        self.n_outputs_ = int(arrays["n_outputs"])
        self.classes_ = arrays["classes"]

        self._offsets = arrays["tree_offsets"]
        self._children_left = arrays["children_left"]
        self._children_right = arrays["children_right"]
        self._feature = arrays["feature"]
        self._threshold = arrays["threshold"]

        # Per-leaf class fractions, from the stored weighted counts. This is
        # sklearn.tree.tree.DecisionTreeClassifier.predict_proba's normaliser,
        # applied once per node here instead of once per predicted row. Same
        # division of the same operands, so the results are bit-identical; the
        # file stays a transcription of the pickle rather than derived numbers.
        value = arrays["value"]
        normalizer = value.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        self._leaf_proba = value / normalizer

    def predict_proba(self, X):
        """Mean of the per-tree class probabilities, over the forest.

        Args:
        X (numpy.ndarray): n_samples x n_features input, any numeric dtype.

        Returns:
        proba (numpy.ndarray): n_samples x n_classes float64, rows summing to 1.
        """
        X = self._validate_X_predict(X)

        proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for t in range(self._n_trees):
            start = int(self._offsets[t])
            stop = int(self._offsets[t + 1])
            leaves = _apply_tree(
                X,
                self._children_left[start:stop],
                self._children_right[start:stop],
                self._feature[start:stop],
                self._threshold[start:stop],
            )
            proba += self._leaf_proba[start:stop][leaves]

        proba /= self._n_trees
        return proba

    def _validate_X_predict(self, X):
        """The subset of sklearn's check_array/_validate_X_predict that can fire here.

        Order matters: sklearn casts and rejects non-finite input before it
        compares the feature count, so a NaN raises ahead of a shape mismatch.
        """
        X = np.asarray(X, dtype=_DTYPE)
        if X.ndim != 2:
            raise ValueError(
                "Expected 2D array, got %dD array instead." % X.ndim
            )
        if not np.isfinite(X).all():
            raise ValueError(
                "Input contains NaN, infinity or a value too large for %r." % X.dtype
            )
        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s " % (self.n_features_, n_features)
            )
        return X


def _apply_tree(X, children_left, children_right, feature, threshold):
    """Return the leaf index reached by each row of X.

    Mirrors ``sklearn.tree._tree.Tree._apply_dense``: descend left when the
    feature value is ``<=`` the threshold, testing ``children_left`` alone for
    the leaf sentinel. The float32 operand promotes to float64 for the
    comparison, as it does in the Cython loop.
    """
    node = np.zeros(X.shape[0], dtype=np.intp)
    rows = np.arange(X.shape[0])
    while True:
        internal = children_left[node] != -1
        if not internal.any():
            break
        idx = rows[internal]
        here = node[idx]
        go_left = X[idx, feature[here]] <= threshold[here]
        node[idx] = np.where(go_left, children_left[here], children_right[here])
    return node


def _load_npz(flip_file):
    with np.load(flip_file, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing = [k for k in _REQUIRED_KEYS if k not in keys]
        if missing:
            raise ValueError(
                "%s is not a flip classifier archive: missing %s"
                % (flip_file, ", ".join(missing))
            )
        arrays = {k: archive[k] for k in _REQUIRED_KEYS}
        metadata = {}
        if "meta_keys" in keys and "meta_values" in keys:
            metadata = dict(
                zip(archive["meta_keys"].tolist(), archive["meta_values"].tolist())
            )

    version = int(arrays["format_version"])
    if version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            "%s has flip classifier format version %d; this version of "
            "moseq2-extract reads %s."
            % (flip_file, version, ", ".join(str(v) for v in _SUPPORTED_FORMAT_VERSIONS))
        )

    n_trees = int(arrays["n_trees"])
    offsets = arrays["tree_offsets"]
    if offsets.shape != (n_trees + 1,):
        raise ValueError(
            "%s: tree_offsets has shape %r, expected (%d,)"
            % (flip_file, offsets.shape, n_trees + 1)
        )
    n_nodes = int(offsets[-1])
    for name in ("children_left", "children_right", "feature", "threshold"):
        if arrays[name].shape != (n_nodes,):
            raise ValueError(
                "%s: %s has shape %r, expected (%d,)"
                % (flip_file, name, arrays[name].shape, n_nodes)
            )
    if arrays["value"].shape != (n_nodes, int(arrays["n_classes"])):
        raise ValueError(
            "%s: value has shape %r, expected (%d, %d)"
            % (flip_file, arrays["value"].shape, n_nodes, int(arrays["n_classes"]))
        )

    return FlipClassifier(arrays, metadata=metadata)


def load_flip_classifier(flip_file):
    """Load a flip classifier from a ``.npz`` archive or a legacy pickle.

    Args:
    flip_file (str): path to the classifier.

    Returns:
    clf: an object with ``classes_``, ``n_features_`` and ``predict_proba``.
    """
    with open(flip_file, "rb") as fh:
        magic = fh.read(len(_NPZ_MAGIC))

    if magic == _NPZ_MAGIC:
        return _load_npz(flip_file)

    import joblib

    return joblib.load(flip_file)


def is_legacy_pickle_failure(exc):
    """Whether this load failed because the file is a scikit-learn 0.19 pickle.

    That failure is the one this port introduces: the old stack read those
    pickles and no modern scikit-learn can. Every other way a load can fail --
    a truncated file, something that is not a classifier at all -- failed the
    same way before the port and must keep failing the same way, so only this
    case is worth reporting differently.
    """
    text = "%s: %s" % (type(exc).__name__, exc)
    return isinstance(exc, (ImportError, ModuleNotFoundError)) and "sklearn" in text

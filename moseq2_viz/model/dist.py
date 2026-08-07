"""
Utility functions for estimating syllable similarity (behavioral distance).
"""

import warnings
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from dtaidistance import dtw_ndim
from cytoolz import keyfilter, curry
from moseq2_viz.util import strided_app, h5_to_dict
from scipy.spatial.distance import squareform, pdist
from moseq2_viz.model.util import (
    whiten_pcs,
    parse_model_results,
    simulate_ar_trajectory,
    get_transitions,
    get_syllable_slices,
    retrieve_pcs_from_slices,
    normalize_pcs,
)
from moseq2_viz.scalars.util import (
    get_scalar_map,
    get_scalar_triggered_average,
    process_scalars,
)


def get_behavioral_distance(
    index,
    model_file,
    whiten="all",
    distances=["ar[init]", "scalars"],
    max_syllable=None,
    resample_idx=-1,
    dist_options={},
    sort_labels_by_usage=True,
    count="usage",
):
    """
    Compute the behavioral distance (square) matrices with respect to a predefined set of variables.

    Args:
    index (str): Path to index file
    model_file (str): Path to trained model
    whiten (str): Indicates whether to whiten all PCs at once or each one at a time. Options = ['all', 'each']
    distances (list or str): type of distance(s) to compute. Available options = ['scalars', 'ar[init]', 'ar[dtw]', 'pca[dtw]', 'combined']
    max_syllable (int): the index of the maximum number of syllables to include
    resample_idx (int): Indicates the parsing method according to the shape of the labels array.
    dist_options (dict): Dictionary holding each distance operations configurable parameters
    sort_labels_by_usage (bool): boolean flag that indicates whether to relabel syllables by count ordering
    count (str): method to compute syllable mean usage, either 'usage' or 'frames'.

    Returns:
    dist_dict (dict): Dictionary containing all computed behavioral square distance matrices
    """

    dist_dict = {}

    defaults = {
        "scalars": {"nlags": 10, "zscore": False},
        "ar[init]": {"sim_points": 10},
        "ar[dtw]": {"sim_points": 60, "parallel": False},
        "pca[dtw]": {
            "normalize": "zscore",
            "max_dur": 30,
            "subsampling": 5,
            "max_samples": None,
            "npcs": 10,
            "remove_offset": False,
            "parallel": False,
        },
        "combined": {
            "combiners": ["pca[dtw]", "scalars"],
            "include_scalars": [
                "velocity_3d_mm",
                "angle",
                "height_ave_mm",
                "width_mm",
                "length_mm",
            ],
        },
    }
    if isinstance(distances, str):
        distances = [distances]

    for k in defaults:
        dist_options[k] = {**defaults[k], **dist_options.get(k, dict())}

    model_fit = parse_model_results(
        model_file,
        resample_idx=resample_idx,
        map_uuid_to_keys=True,
        sort_labels_by_usage=sort_labels_by_usage,
        count=count,
    )

    # make sure the index only uses (a) files that exist and (b) files in the model fit
    # master uuid list...uuid exists in PCA file, model file, and index

    uuid_set = set(model_fit["labels"]) & set(index["files"])

    # only keep animals that were modeled and in the files within the sorted_index
    in_uuid_set = curry(keyfilter)(lambda x: x in uuid_set)
    index["files"] = in_uuid_set(index["files"])
    model_fit["labels"] = in_uuid_set(model_fit["labels"])

    if max_syllable is None:
        max_syllable = -np.inf
        for lbl in model_fit["labels"].values():
            if lbl.max() > max_syllable:
                max_syllable = lbl.max() + 1

    for dist in distances:
        if dist.lower() in ["ar[init]", "ar[dtw]"]:

            ar_mat = model_fit["model_parameters"]["ar_mat"]
            npcs = ar_mat[0].shape[0]
            nlags = ar_mat[0].shape[1] // npcs

            scores = h5_to_dict(index["pca_path"], "scores")

            for k, v in scores.items():
                scores[k] = scores[k][:, :npcs]

            scores = whiten_pcs(scores, whiten)
            init = get_init_points(
                scores,
                model_fit["labels"],
                nlags=nlags,
                npcs=npcs,
                max_syllable=max_syllable,
            )

            if dist.lower() == "ar[init]":
                dist_dict["ar[init]"] = get_behavioral_distance_ar(
                    ar_mat,
                    init_point=init,
                    **dist_options["ar[init]"],
                    max_syllable=max_syllable,
                    dist="correlation",
                )
            elif dist.lower() == "ar[dtw]":
                dist_dict["ar[dtw]"] = get_behavioral_distance_ar(
                    ar_mat,
                    init_point=init,
                    **dist_options["ar[dtw]"],
                    max_syllable=max_syllable,
                    dist="dtw",
                )
        elif dist.lower() == "scalars":
            scalar_map = get_scalar_map(index)
            scalar_ave = get_scalar_triggered_average(
                scalar_map,
                model_fit["labels"],
                max_syllable=max_syllable,
                **dist_options["scalars"],
            )

            if "nlags" in dist_options["scalars"].keys():
                scalar_nlags = dist_options["scalars"]["nlags"]
            else:
                scalar_nlags = None

            for k, v in scalar_ave.items():
                key = f"scalar[{k}]"
                if scalar_nlags is None:
                    scalar_nlags = v.shape[1] // 2
                v = v[:, scalar_nlags + 1 :]
                dist_dict[key] = squareform(pdist(v, "correlation"))

        elif dist.lower() == "pca[dtw]":

            slice_fun = get_syllable_slices(
                labels=list(model_fit["labels"].values()),
                label_uuids=list(model_fit["labels"].keys()),
                index=index,
            )

            pca_scores = h5_to_dict(index["pca_path"], "scores")
            pca_scores = normalize_pcs(
                pca_scores, method=dist_options["pca[dtw]"]["normalize"]
            )
            use_options = deepcopy(dist_options["pca[dtw]"])
            use_options.pop("normalize")
            parallel = use_options.pop("parallel")

            pc_slices = []
            for syllable in tqdm(
                range(max_syllable), desc="Retrieving Syllable Aligned PC Slices"
            ):
                pc_slice = retrieve_pcs_from_slices(
                    slice_fun(syllable), pca_scores, **use_options
                )
                pc_slices.append(pc_slice)

            lens = [_.shape[0] for _ in pc_slices]
            pc_mat = np.concatenate(pc_slices, axis=0)

            # all lengths need to be equal for our current, naive subsampling implementation
            if len(set(lens)) != 1:
                warnings.warn(
                    "Number of example per syllable not equal, returning full matrix"
                )
                dist_dict["pca[dtw]"] = pc_mat
                dist_dict["pca[dtw] (syllables)"] = lens
            else:
                print("Computing DTW matrix (this may take a minute)...")
                full_dist_mat = _pairwise_dtw_matrix(
                    dtw_ndim.distance_matrix(
                        pc_mat, parallel=parallel, show_progress=True
                    )
                )
                reduced_mat = reformat_dtw_distances(full_dist_mat, len(pc_slices))
                dist_dict["pca[dtw]"] = reduced_mat
        elif dist.lower() == "combined":

            npcs = dist_options["pca[dtw]"].get("npcs", 10)
            scalar_map = get_scalar_map(index)
            incl_keys = dist_options["combined"].pop("include_scalars")

            scalar_dict = process_scalars(
                scalar_map,
                include_keys=incl_keys,
                zscore=dist_options["scalars"].get("zscore", False),
            )

            pca_scores = h5_to_dict(index["pca_path"], "scores")
            pca_scores = normalize_pcs(
                pca_scores, method=dist_options["pca[dtw]"]["normalize"]
            )

            pca_scores = {
                k: np.concatenate([v[:, :npcs], scalar_dict[k].T], axis=1)
                for k, v in pca_scores.items()
                if k in scalar_dict
            }

            use_options = deepcopy(dist_options["pca[dtw]"])
            use_options.pop("normalize")
            parallel = use_options.pop("parallel")
            use_options["npcs"] += len(incl_keys)

            slice_fun = get_syllable_slices(
                labels=[model_fit["labels"][k] for k in pca_scores],
                label_uuids=list(pca_scores.keys()),
                index=index,
                trim_nans=False,
            )

            pc_slices = []
            for syllable in tqdm(
                range(max_syllable), desc="Retrieving Syllable Aligned PC Slices"
            ):
                pc_slice = retrieve_pcs_from_slices(
                    slice_fun(syllable), pca_scores, **use_options
                )
                pc_slices.append(pc_slice)

            pc_mat = np.concatenate(pc_slices, axis=0)

            full_dist_mat = _pairwise_dtw_matrix(
                dtw_ndim.distance_matrix(
                    pc_mat, parallel=parallel, show_progress=True
                )
            )
            reduced_mat = reformat_dtw_distances(full_dist_mat, len(pc_slices))
            dist_dict["combined"] = reduced_mat

    return dist_dict


def get_behavioral_distance_ar(
    ar_mat,
    init_point=None,
    sim_points=10,
    max_syllable=40,
    dist="correlation",
    parallel=False,
):
    """
    Compute behavioral distance with respect to the model's AutoRegressive matrices.
    The function Affords either AR trajectory correlation distance, or computing dynamically time-warped trajectory distances.

    Parameters
    ----------
    ar_mat (numpy.ndarray): Trained model AutoRegressive matrices; shape=(max_syllable, npcs, npcs*nlags+1)
    init_point (list): Initial values as a reference point for distance estimation
    sim_points (int): number of time points to simulate
    max_syllable (int): the index of the maximum number of syllables to include
    dist (str): Distance operation to compute. Either 'correlation' or 'dtw'.
    parallel (bool): Boolean flag that indicates whether to use multiprocessing to compute dtw distances.

    Returns
    -------
    ar_dist (2D numpy array): Computed AR trajectory distances for each AR matrix/model state.
    shape=(max_syllable, max_syllable)
    """

    npcs = ar_mat[0].shape[0]

    if init_point is None:
        init_point = [None] * max_syllable

    ar_traj = np.zeros((max_syllable, sim_points, npcs), dtype="float32")

    for i in range(max_syllable):
        ar_traj[i] = simulate_ar_trajectory(
            ar_mat[i], init_point[i], sim_points=sim_points
        )

    if dist.lower() == "correlation":
        ar_dist = squareform(
            pdist(ar_traj.reshape(max_syllable, sim_points * npcs), "correlation")
        )
    elif dist.lower() == "dtw":
        print("Computing DTW matrix (this may take a minute)...")
        ar_dist = _pairwise_dtw_matrix(
            dtw_ndim.distance_matrix(
                ar_traj, parallel=parallel, show_progress=True
            )
        )
        ar_dist = reformat_dtw_distances(
            ar_dist, nsyllables=ar_dist.shape[0], rescale=False
        )
    else:
        raise RuntimeError(f"Did not understand distance {dist}")

    return ar_dist


def get_init_points(pca_scores, model_labels, max_syllable=40, nlags=3, npcs=10):
    """
    Compute initial AR trajectories based on a cumulative average of lagged-PC Scores over nlags.

    Args:
    pca_scores (numpy.ndarray): Loaded PC Scores. Shape=(npcs, nsamples)
    model_labels (list): list of 1D numpy arrays of relabeled/sorted syllable labels
    max_syllable (int): the index of the maximum number of syllables to include
    nlags (int): Number of lagged frames.
    npcs (int): Number of PCs to use in computation.

    Returns:
    syll_average (list): List containing 2D np arrays of average syllable trajectories over a nlag-strided PC scores array.
    """
    # cumulative average of PCs for nlags

    win = int(nlags * 2 + 1)

    if np.mod(win, 2) == 0:
        win = win + 1

    # grab the windows where 0=syllable onset

    syll_average = []
    count = np.zeros((max_syllable,), dtype="int")

    for i in range(max_syllable):
        syll_average.append(np.zeros((win, npcs), dtype="float32"))

    for k, v in pca_scores.items():

        # if k is not used in modeling step, skip
        if k not in model_labels.keys():
            continue
        # get model labels
        labels = model_labels[k]
        seq_array, locs = get_transitions(labels)

        padded_scores = np.pad(
            v, ((win // 2, win // 2), (0, 0)), "constant", constant_values=np.nan
        )

        for i in range(max_syllable):
            hits = locs[np.where(seq_array == i)[0]]

            if len(hits) < 1:
                continue

            count[i] += len(hits)
            for j in range(npcs):
                win_scores = strided_app(padded_scores[:, j], win, 1)
                syll_average[i][:, j] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        syll_average[i] /= count[i].astype("float")

    return syll_average


def _pairwise_dtw_matrix(mat):
    """
    Put a dtaidistance distance matrix into the layout the reformatting expects.

    `reformat_dtw_distances` reads its input as strictly upper triangular, with
    a non-finite entry wherever no distance was computed: it zeroes the
    non-finite entries and symmetrizes with `rmat += rmat.T`, and reads the
    diagonal as the per-syllable scale used to rescale. dtaidistance returns a
    full symmetric matrix with a zero diagonal, which would double every
    distance and collapse that scale to zero.

    Args:
    mat (2D numpy array): square distance matrix from dtaidistance.

    Returns:
    (2D numpy array): the same distances, strictly above the diagonal.
    """

    out = np.array(mat, dtype="float64")
    out[np.tril_indices(out.shape[0])] = np.inf
    return out


def reformat_dtw_distances(full_mat, nsyllables, rescale=True):
    """
    Reduce full (max states) dynamically time-warped PC Score distance matrices to only include dimensions for a total of nsyllables.

    Args:
    full_mat (np.ndarray): DTW distance matrices for all model states/syllables.
    nsyllables (int): Number of syllables to include in truncated DTW distance matrix.
    rescale (bool): Rescale truncated dtw-distance matrices to match output distribution.

    Returns:
    rmat (2D np array): Reformatted-Truncated DTW Distance Matrix; shape = (nsyllables, nsyllables)
    """

    rmat = deepcopy(full_mat)
    rmat[rmat == np.inf] = np.nan

    nsamples = rmat.shape[0] // nsyllables

    if nsamples > 1:
        rmat = rmat.reshape(rmat.shape[0], nsyllables, nsamples)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rmat = np.nanmean(rmat, axis=2)

            rmat = rmat.T
            rmat = rmat.reshape(nsyllables, nsyllables, nsamples)
            rmat = np.nanmean(rmat, axis=2)

    diag_vals = rmat.diagonal()
    rmat[~np.isfinite(rmat)] = 0
    rmat += rmat.T

    nan_rows = np.all(rmat == 0, axis=1)
    rmat[nan_rows, :] = np.nan
    rmat[:, nan_rows] = np.nan

    if rescale:
        for idx, v in np.ndenumerate(rmat):
            ii = diag_vals[idx[0]]
            jj = diag_vals[idx[1]]
            rmat[idx] = v / (np.sqrt(ii * jj) + 1e-12)

    return rmat

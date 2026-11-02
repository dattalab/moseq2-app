"""
ARHMM utility functions
"""

import numpy as np
from tqdm.auto import tqdm
from functools import partial
from cytoolz import valmap, itemmap
from collections import OrderedDict, defaultdict
from jax_moseq.utils import check_for_nans
from moseq2_model.util import save_arhmm_checkpoint, get_loglikelihoods


class ModelNaNError(RuntimeError):
    """The model held NaNs after a Gibbs sweep, so the fit was stopped.

    The message is built from the attributes when it is read, so a caller that
    knows more than ``train_model`` (``learn_model_wrapper`` knows the model
    file that was not written) can set ``dest_file`` before re-raising.
    """

    def __init__(self, nan_messages, sweep, num_iter, last_checkpoint=None,
                 dest_file=None):
        super().__init__()
        self.nan_messages = list(nan_messages)
        self.sweep = sweep
        self.num_iter = num_iter
        self.last_checkpoint = last_checkpoint
        self.dest_file = dest_file

    def __str__(self):
        lines = [
            f"The model fit stopped after sweep {self.sweep} of {self.num_iter}: "
            "the model holds NaNs (values that are not a number) in:",
        ]
        lines += [f"  - {m}" for m in self.nan_messages]
        lines += [
            "Every later sweep would inherit them, so the fit cannot recover and "
            "its syllables would be meaningless.",
            "",
            "No model file was written"
            + (f" to {self.dest_file}." if self.dest_file else "."),
        ]
        if self.last_checkpoint is not None:
            path, sweep = self.last_checkpoint
            lines.append(
                f"The last checkpoint, {path}, saved after sweep {sweep}, holds the "
                "last state of this fit without NaNs."
            )
        else:
            lines.append(
                "No checkpoint of this fit was saved before the NaNs appeared "
                "(checkpoint_freq, --checkpoint-freq on the command line, saves "
                "one every that many sweeps)."
            )
        lines += [
            "",
            "This usually comes from the data: sessions with long stretches where "
            "the mouse is missing from the frame or still, which the extraction "
            "notebook's \"[OPTIONAL] Run Extraction Validation Tests\" cell can flag, "
            "or sessions where the mouse was absorbed into the background image and "
            "barely appears in the extraction.",
            "What to do: check those sessions' extractions (that cell and each "
            "session's extraction movie), leave out the bad ones by removing their "
            "entries from the index file (moseq2-index.yaml; learn-model reads it "
            "only when given -i), then fit again.",
        ]
        return "\n".join(lines)


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def check_model_for_nans(model, sweep, num_iter, last_checkpoint=None):
    """Raise ``ModelNaNError`` if the backend's model state holds any NaN.

    Runs ``jax_moseq.utils.check_for_nans`` on the jax-moseq model dictionary,
    the same check keypoint-moseq makes after each resampling step. A model
    without a ``model_dict`` (a stand-in in a test) is not checked. For a
    per-session state array (first axis one entry per session, such as the
    robust model's ``states/tau``) the message names the sessions holding NaNs.
    """
    state = getattr(model, "model_dict", None)
    if state is None:
        return
    any_nans, nan_info, messages = check_for_nans(state)
    if not any_nans:
        return

    sessions = list(getattr(model, "_session_order", []))
    detailed = []
    for (path, _), message in zip(nan_info, messages):
        value = _leaf(state, path)
        if path and path[0] == "states" and sessions and value.shape[0] == len(sessions):
            per_session = np.isnan(np.asarray(value)).reshape(len(sessions), -1).sum(1)
            named = [f"{sessions[i]} ({n})" for i, n in enumerate(per_session) if n]
            message += (", in session " if len(named) == 1 else ", in sessions ") + ", ".join(named)
        detailed.append(message)
    raise ModelNaNError(detailed, sweep, num_iter, last_checkpoint)


def train_model(
    model,
    num_iter=100,
    checkpoint_freq=None,
    checkpoint_file=None,
    start=0,
    progress_kwargs={},
    train_data=None,
    val_data=None,
    separate_trans=False,
    groups=None,
    verbose=False,
    check_every=2,
    last_checkpoint=None,
):
    """
    Train ARHMM for inputted number of iterations.

    Args:
    model (ARHMM): model object to train
    num_iter (int): total number of resampling iterations
    checkpoint_freq (int): frequency (iterations) to save a checkpoint of the model
    checkpoint_file (str): path to save new checkpoint file
    start (int): starting iteration index used to resume modeling. Default is 0
    progress_kwargs (dict): keyword arguments for progress bar
    train_data (OrderedDict): dict of training data used for getting log-likelihods if verbose is True
    val_data (OrderedDict): dict of validation data used for getting validation log-likelihoods if verbose is True.
    separate_trans (bool): use separated transition matrices for each group
    groups (list): list of groups included in modeling used for getting log-likelihoods if verbose is True
    verbose (bool): get log-likelihoods at `check_every` interval
    check_every (int): frequency (iterations) to record model training/validation log-likelihoods during training
    last_checkpoint (tuple or None): (path, sweep) of the checkpoint the fit resumed from, named if the model goes NaN

    Returns:
    model (ARHMM): trained model.
    log_likelihood (list): list of training log-likelihoods per session after modeling.
    labels (list): list of labels predicted per session after modeling.
    iter_lls (list): list of training log-likelihoods for each `check_every` iteration.
    iter_holls (list): list of held-out log-likelihoods for each `check_every` iteration.
    interrupt (bool): flag to notify the caller of this function if a keyboard interrupt happened

    Raises:
    ModelNaNError: the model held NaNs after a sweep. The progress bar is closed
        and nothing more is fitted; checkpoints already written stay.
    """

    # Checkpointing boolean
    checkpoint = checkpoint_freq is not None

    iter_lls, iter_holls = [], []

    progress = tqdm(range(start, num_iter), **progress_kwargs, desc="Training ARHMM")
    for itr in progress:
        # Resample states, and gracefully return in case of a keyboard interrupt
        try:
            # Positive-definiteness of the residual covariance is maintained by
            # boosting the diagonal before inversion inside the observation
            # distribution, rather than by writing a data-dependent ridge back
            # into the prior term between iterations.
            model.resample_model()
        except KeyboardInterrupt:
            print("Training manually interrupted.")
            print("Returning and saving current iteration of model. ")
            return (
                model,
                model.log_likelihood(),
                get_labels_from_model(model),
                iter_lls,
                iter_holls,
                True,
            )

        # JAX does not raise when a sweep goes numerically wrong; NaNs spread
        # into every later sweep and would be saved with the model.
        try:
            check_model_for_nans(model, itr + 1, num_iter, last_checkpoint)
        except ModelNaNError:
            progress.close()
            raise

        summ_stats = {
            "model": model,
            "groups": groups,
            "train_data": train_data,
            "val_data": val_data,
            "separate_trans": separate_trans,
        }

        if verbose and ((itr + 1) % check_every == 0):
            # Compute and save iteration training and validation log-likelihoods
            train_ll, ho_ll = get_model_summary(**summ_stats)
            iter_lls.append(train_ll)
            if ho_ll is not None:
                iter_holls.append(ho_ll)

        # checkpoint if needed
        if checkpoint and ((itr + 1) % checkpoint_freq == 0):
            last_checkpoint = (
                training_checkpoint(model, itr, checkpoint_file), itr + 1)

    return (
        model,
        model.log_likelihood(),
        get_labels_from_model(model),
        iter_lls,
        iter_holls,
        False,
    )


def training_checkpoint(model, itr, checkpoint_file):
    """
    Format the model checkpoint filename and save the model checkpoint

    Args:
    model (ARHMM): Model object being trained.
    itr (itr): Current modeling iteration.
    checkpoint_file (str): Model checkpoint filename.

    Returns:
    checkpoint_file (str): path the checkpoint was written to.
    """

    # Pack the data to save in checkpoint
    save_data = {
        "iter": itr + 1,
        "model": model,
        "log_likelihoods": model.log_likelihood(),
        "labels": get_labels_from_model(model),
    }

    # Format checkpoint filename
    checkpoint_file = f"{checkpoint_file}-checkpoint_{itr}.arhmm"

    # Save checkpoint
    save_arhmm_checkpoint(checkpoint_file, save_data)
    return checkpoint_file


def get_model_summary(model, groups, train_data, val_data, separate_trans):
    """
    Compute log-likelihood of train_data and val_data (if not None) when verbose is True.

    Args:
    model (ARHMM): model to compute log-likelihoods.
    groups (list): list of session group names.
    train_data (OrderedDict): Ordered dict of training data
    val_data: (OrderedDict or None): Ordered dict of validation/held-out data
    separate_trans (bool): boolean flag that indicates whether to separate log-likelihoods for each group.

    Returns:
    train_ll (float): normalized average training log-likelihoods across all recording sessions.
    val_ll (float): normalized average held-out log-likelihood across all recording sessions.
    """
    # Get train and validation groups
    if groups is not None:
        train_groups, val_groups = groups
    else:
        # if there are no groups, separate trans cannot be True
        separate_trans = False
        train_groups, val_groups = None, None

    # Compute normalized log-likelihoods for each session
    train_ll = get_loglikelihoods(
        model, train_data, train_groups, separate_trans, normalize=True
    )

    # return early if there is no validation data
    if val_data is None:
        return np.mean(train_ll), None

    # Get iteration heldout/validation log-likelihood values
    val_ll = get_loglikelihoods(
        model, val_data, val_groups, separate_trans, normalize=True
    )

    return np.mean(train_ll), np.mean(val_ll)


def get_labels_from_model(model):
    """
    Grab model labels for each training dataset and place them in a list.

    Args:
    model (ARHMM): trained ARHMM

    Returns:
    labels (list): An array of predicted syllable labels for each training session
    """

    labels = [
        np.append(np.repeat(-5, model.nlags), s.stateseq) for s in model.states_list
    ]
    return labels


def apply_model(model, whitening_params, data_dict, metadata, whiten="all"):
    """
    Apply pre-trained model to data_dict. Note that this function might produce unexpected behavior
    if the model was trained using separate transition matrices for different groups of sessions.

    Args:
        model (ARHMM): pre-trained model
        whitening_params (namedtuple or dict): whitening parameters
        data_dict (OrderedDict): data to apply model to
        metadata (dict): metadata for data_dict

    Returns:
        labels (dict): dictionary of labels predicted per session after modeling
    """

    # whiten data function
    mu, L, offset = (
        whitening_params["mu"],
        whitening_params["L"],
        whitening_params["offset"],
    )
    apply_whitening = lambda x: np.linalg.solve(L, (x - mu).T).T + offset

    # check for whiten parameters to see if whiten_all or whiten_each
    if whiten[0].lower() == "e":
        # this approach is not recommended, but supported
        center = whitening_params[list(whitening_params)[0]]["offset"] == 0
        whitened_data, _ = whiten_each(data_dict, center)
    else:
        whitened_data = valmap(apply_whitening, data_dict)

    # apply model to data
    if "SeparateTrans" in str(type(model)):
        # not recommended, but supported
        labels = itemmap(
            lambda item: (
                item[0],
                model.heldout_viterbi(item[1], group_id=metadata["groups"][item[0]]),
            ),
            whitened_data,
        )
    else:
        labels = valmap(model.heldout_viterbi, whitened_data)

    return labels


# taken from moseq by @mattjj and @alexbw
def whiten_all(data_dict, center=True, fit_on=None):
    """
    Whiten the PC Scores (with Cholesky decomposition) using all the data to compute the covariance matrix.

    Args:
    data_dict (OrderedDict): Training dataset
    center (bool): Indicates whether to center data by subtracting the mean PC score.
    fit_on (OrderedDict): the data the mean and covariance are estimated from, when only part of data_dict
        should shape the transform (the training data, with held-out data present); defaults to data_dict.

    Returns:
    data_dict (OrderedDict): Whitened training data dictionary
    """

    non_nan = lambda x: x[~np.isnan(np.reshape(x, (x.shape[0], -1))).any(1)]
    meancov = lambda x: (x.mean(0), np.cov(x, rowvar=False, bias=1))
    contig = partial(np.require, dtype=np.float64, requirements="C")

    fit_on = data_dict if fit_on is None else fit_on
    mu, Sigma = meancov(np.concatenate(list(map(non_nan, fit_on.values()))))
    L = np.linalg.cholesky(Sigma)

    offset = 0.0 if center else mu
    # set up function to whiten data
    apply_whitening = lambda x: np.linalg.solve(L, (x - mu).T).T + offset
    whitening_parameters = {"mu": mu, "L": L, "offset": offset}
    return (
        OrderedDict((k, contig(apply_whitening(v))) for k, v in data_dict.items()),
        whitening_parameters,
    )


# taken from moseq by @mattjj and @alexbw
def whiten_each(data_dict, center=True):
    """
    Whiten the PC scores for each training dataset separately.

    Args:
    data_dict (OrderedDict): Training dataset
    center (bool): Boolean flag that indicates whether to center data by subtracting the mean PC score.

    Returns:
    data_dict (OrderedDict): Whitened training data dictionary
    """
    whitening_parameters = {}
    for k, v in data_dict.items():
        tmp_dict, whitening_parameters[k] = whiten_all({k: v}, center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict, whitening_parameters


def run_e_step(arhmm):
    """
    Compute the expected state sequence for sessions in the training dataset and place them in a list.

    Args:
    arhmm (ARHMM): model to compute expected states from.

    Returns:
    e_states (list): list of expected states
    """

    arhmm._E_step()
    return [s.expected_states for s in arhmm.states_list]


def zscore_each(data_dict, center=True):
    """
    z-score each set of PC Scores separately

    Args:
    data_dict (OrderedDict): Training dictionary
    center (bool): Indicates whether to center data by subtracting the mean PC score.

    Returns:
    data_dict (OrderedDict): z-scored training data dictionary
    """

    for k, v in data_dict.items():
        tmp_dict = zscore_all({k: v}, center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict


def zscore_all(data_dict, npcs=10, center=True):
    """
    z-score the PC Scores altogether.

    Args:
    data_dict (OrderedDict): Training dictionary
    npcs (int): number of pcs included
    center (bool): Indicates whether to center data by subtracting the mean PC score.

    Returns:
    data_dict (OrderedDict): z-scored training data dictionary
    """

    scores = np.concatenate(list(data_dict.values()))[:, :npcs]
    mu, sig = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)

    for k, v in data_dict.items():
        numerator = v[:, :npcs] - mu if center else v[:, :npcs]
        data_dict[k] = numerator / sig

    return data_dict


# taken from syllables by @alexbw
def get_crosslikes(arhmm, frame_by_frame=False):
    """
    Get the cross-likelihoods, a measure of confidence in label segmentation, for each model label.

    Args:
    arhmm: the ARHMM model object
    frame_by_frame (bool): if True, the cross-likelihoods will be computed for each frame.

    Returns:
    All_CLs (list): a dictionary containing cross-likelihoods for each syllable pair.
    CL (np.ndarray): the average cross-likelihood for each syllable pair
    """

    all_CLs = defaultdict(list)
    Nstates = arhmm.num_states

    if frame_by_frame:
        # Optionally compute Cross-state log-likelihoods over all frame
        for s in arhmm.states_list:
            for j in range(Nstates):
                likes = s.aBl[s.stateseq == j]
                for i in range(Nstates):
                    all_CLs[(i, j)].append(likes[:, i] - likes[:, j])
        all_CLs = valmap(np.concatenate, all_CLs)
    else:
        # Compute cross log-likelihoods across all modeling states
        for s in arhmm.states_list:
            for j in range(Nstates):
                for sl in slices_from_indicators(s.stateseq == j):
                    likes = np.nansum(s.aBl[sl], axis=0)
                    for i in range(Nstates):
                        all_CLs[(i, j)].append(likes[i] - likes[j])

    # Pack cross log-likelihoods into a square confusion matrix
    CL = np.zeros((Nstates, Nstates))
    for (i, j), _ in np.ndenumerate(CL):
        CL[i, j] = np.nanmean(all_CLs[(i, j)])

    return all_CLs, CL


def slices_from_indicators(indseq):
    """
    Compute start and stop indices (slices) for each contiguous sequence of True values in `indseq`.

    Args:
    indseq (list): Indicator array, containing True and False values

    Returns:
    (list): list of slices from `indseq`.
    """

    return [sl for sl in rleslices(indseq) if indseq[sl.start]]


def rleslices(seq):
    """
    Get changepoint slices

    Args:
    seq (list): list of labels

    Returns:
    (map generator): slices of syllable changepoints
    """

    (pos,) = np.where(np.diff(seq) != 0)
    pos = np.concatenate(([0], pos + 1, [len(seq)]))
    return map(slice, pos[:-1], pos[1:])

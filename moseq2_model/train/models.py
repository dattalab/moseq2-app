"""
ARHMM initialization utilities.
"""

import warnings
import numpy as np
from cytoolz import merge
from functools import partial
from moseq2_model.train import backend


flush_print = partial(print, flush=True)


def _ar_statistics(train_datas, nlags, affine, data_dim):
    """Sufficient statistics of one autoregressive model fitted to all data.

    Returns ``(yyT, yxT, xxT, n)`` for the strided design, with the affine
    coordinate appended to the predictors when ``affine`` is set. Rows holding
    a NaN anywhere are dropped, so a session with missing frames contributes
    only its complete lag windows.
    """
    ar_dim = data_dim * nlags + int(affine)
    xxT = np.zeros((ar_dim, ar_dim))
    yxT = np.zeros((data_dim, ar_dim))
    yyT = np.zeros((data_dim, data_dim))
    n = 0

    for data in train_datas:
        rows = backend.ar_striding(np.asarray(data, dtype=np.float64), nlags)
        rows = rows[~np.isnan(rows).any(axis=1)]
        if rows.shape[0] == 0:
            continue

        x, y = rows[:, :-data_dim], rows[:, -data_dim:]
        if affine:
            x = np.hstack((x, np.ones((rows.shape[0], 1))))

        xxT += x.T @ x
        yxT += y.T @ x
        yyT += y.T @ y
        n += rows.shape[0]

    return yyT, yxT, xxT, n


# Empirical bayes estimate of S_0 (from MoSeq)
def _get_empirical_ar_params(train_datas, params, nlags):
    """
    Estimate the parameters of an AR observation model by fitting a single AR model to the entire dataset.

    Args:
    train_datas (list): list of np.ndarrays representing each session's PC scores
    params (dict): dict containing modeling parameters used for initialization
    nlags (int): number of lag frames in the autoregressive design

    Returns:
    obs_params (dict): dict of observed parameters to use in modeling.
    """

    assert isinstance(train_datas, list) and len(train_datas) > 0
    datadimension = train_datas[0].shape[1]
    assert params["nu_0"] > datadimension + 1

    # Initialize the observation parameters
    obs_params = dict(
        nu_0=params["nu_0"],
        S_0=params["S_0"],
        M_0=params["M_0"],
        K_0=params["K_0"],
        affine=params["affine"],
    )

    # Fit an AR model to the entire dataset by maximum likelihood
    yyT, yxT, xxT, n = _ar_statistics(
        train_datas, nlags, params["affine"], datadimension
    )
    A = np.linalg.solve(xxT, yxT.T).T
    sigma = (yyT - A @ yxT.T) / n
    # Symmetrize and lift off the boundary, as the closed-form solution is only
    # symmetric and positive definite up to rounding.
    sigma = 1e-10 * np.eye(datadimension) + (sigma + sigma.T) / 2.0

    # Use the inferred noise covariance as the prior mean
    # E_{IW}[S] = S_0 / (nu_0 - datadimension - 1)
    obs_params["S_0"] = sigma * (params["nu_0"] - datadimension - 1)

    return obs_params


def ARHMM(
    data_dict,
    kappa=1e6,
    gamma=999,
    nlags=3,
    alpha=5.7,
    K_0_scale=10.0,
    S_0_scale=0.01,
    max_states=100,
    empirical_bayes=True,
    affine=True,
    model_hypparams={},
    obs_hypparams={},
    sticky_init=False,
    separate_trans=False,
    groups=None,
    robust=False,
    silent=False,
    seed=None,
):
    """
    Initialize ARHMM and add data and group labels to the ARHMM.

    Args:
    data_dict (OrderedDict): training data to add to model
    kappa (float): hyperparameter for setting syllable duration. Larger kappa = longer syllable durations
    gamma (float): scaling parameter for hierarchical dirichlet process (try to not change it)
    nlags (int): number of lag frames to add to sessions
    alpha (float): scaling parameter for hierarchical dirichlet process (try to not change it)
    K_0_scale (float): Standard deviation of lagged data
    S_0_scale (float): scale standard deviation initialization (try to not change it)
    max_states (int): Maximum number of model states
    empirical_bayes (bool): Boolean flag that specifies using empirical bayes to initialize sigma
    affine (bool): Boolean flag that specifies using affine transformation in the AR processes
    model_hypparams (dict): other model parameters (try to not change it)
    obs_hypparams (dict): observed parameters nu_0, S_0, M_0, and K_0 (try to not change it)
    sticky_init (bool): accepted for backwards compatibility and ignored
    separate_trans (bool): Boolean flag that specifies using separate transition matrices for each group
    groups (list): list of groups to model
    robust (bool): Boolean flag that specifies using student's t-distributed AR model
    silent (bool): Boolean flag that specifies printing out model information
    seed (int): seed for the sampler. Drawn from numpy's global generator when
        unset, so ``np.random.seed`` upstream still determines the fit.

    Returns:
    model (JaxARHMM): initialized model object
    """

    # numpy moved VisibleDeprecationWarning under np.exceptions in 1.25 and
    # dropped the top-level alias in 2.0. The warning it silenced came from
    # pyhsmm building ragged object arrays out of the per-session data; nothing
    # on the jax-moseq path does that, so the filter is kept only so a caller
    # that was relying on the suppression does not start seeing output.
    warnings.filterwarnings(
        "ignore", category=np.exceptions.VisibleDeprecationWarning
    )

    # Get dimension of training data
    data_dim = list(data_dict.values())[0].shape[1]

    # Set observational hyper-parameters
    default_obs_hypparams = {
        "nu_0": data_dim + 2,
        "S_0": S_0_scale * np.eye(data_dim),
        # The identity block sits on the most recent lag, giving a random-walk
        # prior mean. The autoregressive design vector is ordered oldest lag
        # first (row t is [x_{t-nlags}, ..., x_{t-1}]), so the identity belongs
        # in the final lag block, before the affine coordinate.
        "M_0": np.hstack(
            (
                np.zeros((data_dim, data_dim * (nlags - 1))),
                np.eye(data_dim),
                np.zeros((data_dim, int(affine))),
            )
        ),
        "affine": affine,
        "K_0": K_0_scale * np.eye(data_dim * nlags + affine),
    }

    # Set modeling hyper-parameters
    default_model_hypparams = {
        "alpha": alpha,
        "gamma": gamma,
        "kappa": kappa,
        "init_state_distn": "uniform",
    }

    # Update hyper-parameters given function inputted dict objects
    obs_hypparams = merge(default_obs_hypparams, obs_hypparams)
    model_hypparams = merge(default_model_hypparams, model_hypparams)

    # Use empirical_bayes to set initial sigma values. This is preprocessing
    # rather than part of the sampler, and it runs over every supplied session,
    # including any that separate transition matrices later exclude.
    if empirical_bayes:
        obs_hypparams = _get_empirical_ar_params(
            list(data_dict.values()), obs_hypparams, nlags
        )

    # Sessions whose group is "n/a" are not modelled under separate transition
    # matrices, so they are left out of the data entirely.
    if separate_trans:
        modelled = type(data_dict)(
            (name, data) for name, data in data_dict.items()
            if groups[name] != "n/a"
        )
    else:
        modelled = data_dict

    if not silent:
        flush_print(
            "Using {}model class {}".format(
                "ROBUST " if robust else "",
                "Corrected{}ARHMM{}".format(
                    "Robust" if robust else "Fast",
                    "SeparateTrans" if separate_trans else "",
                ),
            )
        )
        for name in modelled:
            flush_print(f"Adding data from key {name}")
            if separate_trans:
                flush_print(f"Group ID: {groups[name]}")

    print(f"Total number of included sessions: {len(modelled)}")

    return backend.build(
        robust=robust,
        separate_trans=separate_trans,
        data_dict=modelled,
        obs_hypparams=obs_hypparams,
        model_hypparams=model_hypparams,
        max_states=max_states,
        nlags=nlags,
        groups=groups if separate_trans else None,
        seed=np.random.randint(2 ** 31) if seed is None else seed,
    )

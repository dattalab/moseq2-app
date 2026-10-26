"""jax-moseq fitting behind the model interface moseq2-model already uses.

``JaxARHMM`` wraps a jax-moseq model dictionary and exposes the handful of
attributes the rest of moseq2-model reads off a ``pyhsmm`` model. Everything
downstream keeps working unchanged: in particular
``moseq2_model.util.get_parameters_from_model`` is not modified, and the
``model_parameters`` block it produces -- the only part of a saved fit that
moseq2-viz and moseq2-app actually read -- comes out with the same keys, shapes
and dtypes as before.

The attributes are not invented. Each one is here because something reads it:

``nlags``                       ``get_parameters_from_model``, ``train_model``
``obs_distns[k].A/.sigma/.nu``  ``get_parameters_from_model`` -> ``ar_mat``,
                                ``sig``, ``nu``
``trans_distn``                 ``get_parameters_from_model`` -> ``kappa``,
``trans_distns``                ``gamma``, ``alpha``, ``max_states``; the
                                per-group attribute exists only for
                                separate-transition fits, matching the
                                ``hasattr(model, "trans_distns")`` branch
``init_emission_distn``         ``get_parameters_from_model`` -> ``nu_0``,
                                ``sigma_0``, ``kappa_0``, ``mu_0``
``states_list[i].stateseq``     ``get_labels_from_model``, ``train_model``
``log_likelihood``              ``get_loglikelihoods``
``resample_model``              ``train_model``

``__class__.__name__`` feeds ``model_parameters["model_class"]``, so the
concrete subclasses below carry the names the pyhsmm classes had rather than
their own.

Sessions are handed to jax-moseq as one rectangular array. moseq2 supplies
per-session arrays of unequal length, so they are padded to the longest and the
padding is marked invalid, which is the same path a dropped frame already
takes.
"""

import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import jax
import numpy as np

# Every equivalence result to date, and moseq2-model's robust path before the
# swap, is double precision; jax-moseq defaults to single. This is a
# process-wide setting, so importing moseq2-model changes JAX's precision for
# anything else running in the same interpreter.
jax.config.update("jax_enable_x64", True)

na = np.newaxis


def ar_striding(data, nlags):
    """Rows of ``[x_t, ..., x_{t+nlags}]``, oldest lag first.

    The layout ``autoregressive.util.AR_striding`` produced, reimplemented so
    the legacy package can leave the dependency tree. The target is the final
    ``latent_dim`` columns and the predictors are everything before it.
    """
    a = np.ascontiguousarray(data)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    n_frames, dim = a.shape
    itemsize = a.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        a,
        shape=(n_frames - nlags, dim * (nlags + 1)),
        strides=(dim * itemsize, itemsize),
    )


class _ObsDistn(object):
    """One state's observation parameters, in the shape moseq2 reads them.

    ``nu`` is present only for the robust model, mirroring
    ``get_parameters_from_model``'s ``if "nu" in dir(model.obs_distns[0])``
    test, which is how it decides whether to record degrees of freedom.
    """

    __slots__ = ("A", "sigma", "nu")

    def __init__(self, A, sigma, nu=None):
        self.A = A
        self.sigma = sigma
        if nu is not None:
            self.nu = nu

    def __dir__(self):
        base = ["A", "sigma"]
        return base + ["nu"] if hasattr(self, "nu") else base


class _TransDistn(object):
    """One transition distribution's hyperparameters."""

    __slots__ = ("kappa", "gamma", "alpha", "N", "trans_matrix", "beta")

    def __init__(self, kappa, gamma, alpha, num_states, trans_matrix, betas):
        self.kappa = kappa
        self.gamma = gamma
        self.alpha = alpha
        self.N = num_states
        self.trans_matrix = trans_matrix
        self.beta = betas


class _TransDistns(dict):
    """Per-group transition distributions, keyed by group label.

    ``get_parameters_from_model`` reads a separate-transition fit's
    hyperparameters as ``model.trans_distns[0]``, an integer index rather than
    a group label. pyhsmm stores these in a ``defaultdict``, so that lookup
    quietly manufactures an entry instead of failing, and the values it goes on
    to read -- ``kappa``, ``gamma``, ``alpha`` and the state count -- are shared
    across groups anyway. Any absent key therefore resolves to the first
    group's distribution, which carries the same hyperparameters, without
    growing the mapping.
    """

    def __missing__(self, key):
        if not self:
            raise KeyError(key)
        return next(iter(self.values()))


class _InitEmissionDistn(object):
    """Stand-in for ``autoregressive``'s initial-emission Gaussian.

    That object exists only to generate the first ``nlags`` frames when a model
    is asked to synthesize data, and its hyperparameters are fixed by the
    dimensions rather than learned:
    ``Gaussian(nu_0=P+1, sigma_0=10*P*I, mu_0=0, kappa_0=1)`` with
    ``P = latent_dim * nlags`` (``autoregressive/models.py:20``). Reproducing
    them keeps ``model_parameters`` byte-comparable with a pyhsmm fit.
    """

    def __init__(self, latent_dim, nlags):
        p = latent_dim * nlags
        self.hypparams = {
            "nu_0": p + 1,
            "sigma_0": 10 * p * np.eye(p),
            "mu_0": np.zeros(p),
            "kappa_0": 1.0,
        }


class _States(object):
    """One session's sampled state sequence.

    ``expected_states`` is filled by :py:meth:`JaxARHMM._E_step` and is None
    until then, matching a pyhsmm states object before its first E step.
    """

    # ``data`` is the strided training data on a pyhsmm states object, which
    # ``copy_model`` blanks before saving. Nothing is stored here, but the
    # attribute has to be settable for that to run.
    __slots__ = ("stateseq", "group_id", "data", "expected_states",
                 "_owner", "_index")

    def __init__(self, stateseq, group_id=None, owner=None, index=None):
        self.stateseq = stateseq
        self.group_id = group_id
        self.data = None
        self.expected_states = None
        self._owner = owner
        self._index = index

    @property
    def aBl(self):
        """Log likelihood of each frame under each state, ``(T - nlags, K)``.

        Computed on demand and cached on the model rather than built for every
        session up front: at production scale one session's matrix is around
        30 MB, and ``get_crosslikes`` is the only caller.
        """
        if self._owner is None:
            raise AttributeError("states object is not attached to a model")
        return self._owner._session_aBl(self._index)


class JaxARHMM(object):
    """An AR-HMM fitted by jax-moseq, presented as a moseq2 model object.

    Parameters
    ----------
    data_dict : OrderedDict
        Session name to PC scores, one array per session, lengths may differ.
    obs_hypparams : dict
        ``nu_0``, ``S_0``, ``M_0``, ``K_0``, ``affine``. ``S_0`` is whatever the
        caller resolved, including the empirical-Bayes estimate, which stays in
        moseq2 because it is preprocessing rather than part of the fitter.
    model_hypparams : dict
        ``alpha``, ``gamma``, ``kappa``.
    robust : bool
        Student-t observation noise.
    groups : dict or None
        Session name to group label. Present only for separate-transition fits.
    seed : int
        Seed for the jax-moseq PRNG.
    """

    _model_class_name = "ARWeakLimitStickyHDPHMM"

    def __init__(self, data_dict, obs_hypparams, model_hypparams,
                 max_states=100, nlags=3, robust=False, groups=None, seed=0):
        import jax.numpy as jnp
        import jax.random as jr

        self.robust = robust
        self.nlags = nlags
        self.num_states = max_states
        self._session_order = list(data_dict)
        self._groups = groups

        first = next(iter(data_dict.values()))
        self.latent_dim = first.shape[1]

        x, mask, valid = _stack_sessions(data_dict, nlags)
        self._data = {"x": jnp.asarray(x), "mask": jnp.asarray(mask.astype(float))}
        # Autoregressive rows per session, which is what a state sequence is
        # indexed by. Dropped frames still occupy a row and still receive a
        # state; they are excluded from the statistics, not from the sequence.
        self._lengths = [
            np.asarray(data_dict[k]).shape[0] - nlags
            for k in self._session_order
        ]
        self._valid = valid
        self._clear_derived()

        if groups is not None:
            codes = sorted({groups[k] for k in self._session_order})
            self._group_codes = codes
            self._data["group"] = jnp.asarray(
                np.array([codes.index(groups[k]) for k in self._session_order])
            )

        hyp = self._build_hypparams(obs_hypparams, model_hypparams)
        self._model = self._module.init_model(
            data=self._data, seed=jr.PRNGKey(seed), **hyp
        )
        self.init_emission_distn = _InitEmissionDistn(self.latent_dim, nlags)

    # -- construction helpers ------------------------------------------------

    @property
    def _module(self):
        """The jax-moseq model module this fit runs on.

        Resolved on each access rather than stored, so no module object lands
        in the instance dictionary. ``copy_model`` deep-copies the model before
        saving it, and a module cannot be deep-copied.
        """
        from jax_moseq.models import arhmm, robust_arhmm

        base = robust_arhmm if self.robust else arhmm
        return base.separate_trans if self._groups is not None else base

    def _build_hypparams(self, obs_hypparams, model_hypparams):
        """Translate moseq2's hyperparameters into jax-moseq's layout.

        ``S_0``, ``M_0`` and ``K_0`` are passed through explicitly rather than
        rebuilt from scales, so an empirical-Bayes ``S_0`` survives.
        """
        import jax.numpy as jnp

        trans = {
            "num_states": self.num_states,
            "alpha": model_hypparams["alpha"],
            "gamma": model_hypparams["gamma"],
            "kappa": model_hypparams["kappa"],
        }
        ar = {
            "latent_dim": self.latent_dim,
            "nlags": self.nlags,
            "num_states": self.num_states,
            "nu_0": float(obs_hypparams["nu_0"]),
            "S_0": jnp.asarray(obs_hypparams["S_0"]),
            "M_0": jnp.asarray(obs_hypparams["M_0"]),
            "K_0": jnp.asarray(obs_hypparams["K_0"]),
        }
        out = {"hypparams": {"trans_hypparams": trans, "ar_hypparams": ar}}
        if self._groups is not None:
            out["num_groups"] = len(self._group_codes)
        return out

    # -- the surface the rest of moseq2-model reads --------------------------

    @property
    def obs_distns(self):
        """One entry per state, rebuilt when the parameters change.

        Held rather than reconstructed on each access, because pyhsmm's
        ``obs_distns`` was a stable list and callers rely on that: reading
        ``[o.A for o in model.obs_distns]`` twice gave the same array objects,
        so a caller comparing the two got identity rather than an elementwise
        array comparison. The cache is dropped by ``_clear_derived`` whenever a
        sweep moves the parameters, so it can never go stale.
        """
        if self._obs_distns is None:
            params = self._model["params"]
            Ab = np.asarray(params["Ab"])
            Q = np.asarray(params["Q"])
            nu = np.asarray(params["nu"]) if self.robust else None
            self._obs_distns = [
                _ObsDistn(Ab[k], Q[k], None if nu is None else float(nu[k]))
                for k in range(self.num_states)
            ]
        return self._obs_distns

    @property
    def trans_distn(self):
        if self._groups is not None:
            raise AttributeError(
                "separate-transition fits expose trans_distns, not trans_distn"
            )
        return self._trans_distn_for(None)

    @property
    def trans_distns(self):
        if self._groups is None:
            raise AttributeError(
                "single-transition fits expose trans_distn, not trans_distns"
            )
        return _TransDistns(
            (g, self._trans_distn_for(i))
            for i, g in enumerate(self._group_codes)
        )

    def _trans_distn_for(self, index):
        trans = self._model["hypparams"]["trans_hypparams"]
        pi = np.asarray(self._model["params"]["pi"])
        betas = np.asarray(self._model["params"]["betas"])
        if index is not None:
            pi, betas = pi[index], betas[index]
        return _TransDistn(
            float(trans["kappa"]), float(trans["gamma"]), float(trans["alpha"]),
            self.num_states, pi, betas,
        )

    @property
    def states_list(self):
        z = np.asarray(self._model["states"]["z"])
        out = []
        for i, name in enumerate(self._session_order):
            group = None if self._groups is None else self._groups[name]
            states = _States(
                z[i, : self._lengths[i]].astype("int32"), group,
                owner=self, index=i,
            )
            if self._expected_states is not None:
                states.expected_states = self._expected_states[i, : self._lengths[i]]
            out.append(states)
        return out

    @property
    def stateseqs(self):
        return [s.stateseq for s in self.states_list]

    def resample_model(self, num_procs=0, **kwargs):
        """One Gibbs sweep. ``num_procs`` is accepted and ignored."""
        self._model = self._module.resample_model(self._data, **self._model)
        self._clear_derived()

    # -- quantities derived from the current parameters ----------------------

    def _clear_derived(self):
        """Drop everything that a resampling step invalidates."""
        self._aBl_cache = {}
        self._expected_states = None
        self._obs_distns = None

    def _state_log_likelihoods(self, x):
        """Log likelihood of each frame of ``x`` under every state.

        Returns an array shaped ``(..., T - nlags, num_states)``, the state
        axis last, which is the layout ``aBl`` is indexed in.
        """
        import jax
        import jax.numpy as jnp

        params = self._model["params"]
        if self.robust:
            from jax_moseq.models.robust_arhmm.log_prob import (
                robust_ar_log_likelihood,
            )

            fn, args = robust_ar_log_likelihood, (
                params["Ab"], params["Q"], params["nu"])
        else:
            from jax_moseq.utils.autoregression import ar_log_likelihood

            fn, args = ar_log_likelihood, (params["Ab"], params["Q"])

        return jnp.moveaxis(jax.lax.map(partial(fn, x), args), 0, -1)

    def _session_aBl(self, index):
        """One session's ``aBl``, computed on first access and cached.

        Frames excluded from the fit are set to NaN rather than scored. Their
        latent values were zero-filled to keep the array rectangular, so a
        finite likelihood there would be an artefact of the padding.
        ``get_crosslikes`` reduces with ``nansum`` and ``nanmean``, which is the
        same treatment a NaN frame received before the swap.
        """
        cached = self._aBl_cache.get(index)
        if cached is not None:
            return cached

        rows = self._lengths[index]
        aBl = np.asarray(
            self._state_log_likelihoods(self._data["x"][index][na])
        )[0][:rows]
        aBl = np.array(aBl, dtype=np.float64)
        aBl[~self._valid[index, :rows]] = np.nan

        self._aBl_cache[index] = aBl
        return aBl

    def _E_step(self):
        """Fill each session's posterior over states at every frame.

        The counterpart of pyhsmm's ``_E_step``, which ``run_e_step`` calls
        before reading ``expected_states`` off the states objects. Under
        separate transition matrices the sessions are grouped, since each
        group's forward-backward pass runs against its own transition matrix.
        """
        module = self._base_module()
        params = self._model["params"]
        x, mask = self._data["x"], self._data["mask"]
        extra = {"nu": params["nu"]} if self.robust else {}

        def marginals(x_part, mask_part, pi):
            return np.asarray(module.stateseq_marginals(
                x_part, mask_part, params["Ab"], params["Q"], pi=pi, **extra
            ))

        if self._groups is None:
            out = marginals(x, mask, params["pi"])
        else:
            codes = np.asarray(self._data["group"])
            out = np.zeros(
                (x.shape[0], x.shape[1] - self.nlags, self.num_states))
            for g in range(len(self._group_codes)):
                rows = np.flatnonzero(codes == g)
                if rows.size:
                    out[rows] = marginals(x[rows], mask[rows], params["pi"][g])

        self._expected_states = out
        return out

    def attach_data(self, data_dict):
        """Restore the training data to a model loaded from a checkpoint.

        ``copy_model`` strips the data before a checkpoint is written, so a
        reloaded model can report its parameters but cannot resample until the
        arrays are put back. The sessions must be the ones the fit ran on:
        resuming against different data would continue a chain whose state
        sequences no longer describe it.
        """
        import jax.numpy as jnp

        missing = [k for k in self._session_order if k not in data_dict]
        if missing:
            raise ValueError(
                "checkpoint expects sessions absent from the training data: "
                + ", ".join(map(str, missing[:5]))
            )

        ordered = OrderedDict((k, data_dict[k]) for k in self._session_order)
        lengths = [np.asarray(v).shape[0] - self.nlags for v in ordered.values()]
        if lengths != self._lengths:
            raise ValueError(
                "session lengths do not match the checkpoint: expected %r, got %r"
                % (self._lengths[:5], lengths[:5])
            )

        x, mask, valid = _stack_sessions(ordered, self.nlags)
        self._data["x"] = jnp.asarray(x)
        self._data["mask"] = jnp.asarray(mask.astype(float))
        self._valid = valid
        self._clear_derived()
        return self

    def log_likelihood(self, data=None, group_id=None):
        """Marginal log likelihood, with the discrete states summed out.

        With no argument this is the whole fitted dataset, which is what
        ``train_model`` records. Given one session's array it is evaluated for
        that session alone, by treating it as a batch of one; ``get_loglikelihoods``
        calls it that way once per session and then normalizes by length, so an
        apportioned total would give every session the same number.
        """
        params = self._model["params"]
        if data is None:
            x, mask = self._data["x"], self._data["mask"]
        else:
            x, mask = self._one_session(data)

        args = dict(mask=mask, x=x, Ab=params["Ab"], Q=params["Q"],
                    pi=self._pi_for(group_id))
        if self.robust:
            args["nu"] = params["nu"]
        return float(self._base_module().marginal_log_likelihood(**args))

    def heldout_viterbi(self, data, group_id=None, **kwargs):
        """Most probable state sequence for one session under the fitted model.

        The counterpart of pyhsmm's ``heldout_viterbi``, which ``apply_model``
        calls once per session when a trained model is applied to new
        recordings. Two properties of the original are load-bearing and are
        preserved here.

        It leaves the model untouched. pyhsmm builds a temporary states object
        and pops it straight back off ``states_list``; nothing is written to
        ``self._data`` or ``self._model`` here either, so the same model can be
        applied to any number of sessions in any order.

        It works on a model that was loaded from disk. ``copy_model`` strips the
        training arrays before a fit is saved, so the object ``apply_model``
        receives has parameters but no data. The sequence is therefore built
        from the supplied array alone.

        ``**kwargs`` is accepted and ignored: pyhsmm forwarded its extra
        arguments to ``add_data``, and callers may still pass them.

        Parameters
        ----------
        data : array of shape (T, latent_dim)
            One session's PC scores, already whitened by the caller.
        group_id : hashable, optional
            Group label, for a fit with separate transition matrices.

        Returns
        -------
        stateseq : np.ndarray of shape (T - nlags,), int32
        """
        params = self._model["params"]
        x, mask = self._one_session(data)

        args = dict(x=x, mask=mask, Ab=params["Ab"], Q=params["Q"],
                    pi=self._pi_for(group_id))
        if self.robust:
            args["nu"] = params["nu"]
        return np.asarray(self._base_module().stateseq_mode(**args))[0].astype(
            "int32")

    def _pi_for(self, group_id):
        """Transition matrix for one group, or the shared one.

        A group the fit never saw resolves to the first group's transitions
        rather than raising. This comes up in ``apply_model``: when the PC score
        file carries no group information, ``load_pcs`` falls back to numbering
        the sessions, so a model fitted on named groups is asked for group ``0``.
        pyhsmm stored these in a ``defaultdict`` and silently manufactured a
        fresh transition distribution *drawn from the prior*, decoding against
        unfitted parameters. Falling back to a fitted group is the same control
        flow with a better answer, and it is what ``_TransDistns`` already does.
        """
        pi = self._model["params"]["pi"]
        if self._groups is None:
            return pi
        if group_id is None:
            # No group named: use the first group's transitions, which is what
            # pyhsmm's default group_id resolves to.
            return pi[0]
        try:
            return pi[self._group_codes.index(group_id)]
        except ValueError:
            warnings.warn(
                "group {!r} was not present when this model was fitted; "
                "decoding under group {!r} instead. Supply an index file so "
                "the sessions carry the groups the model was trained on."
                .format(group_id, self._group_codes[0])
            )
            return pi[0]

    def _one_session(self, data):
        """One session's array as the rectangular batch of one jax-moseq wants."""
        import jax.numpy as jnp

        arr = np.asarray(data, dtype=np.float64)
        x, mask, _ = _stack_sessions({"_": arr}, self.nlags)
        return jnp.asarray(x), jnp.asarray(mask.astype(float))

    def _base_module(self):
        """The jax-moseq model module, which re-exports gibbs and log_prob.

        Not the separate-transition submodule: the per-group functions take a
        stacked ``pi`` and a group index, whereas everything reached through
        here is handed one group's transition matrix directly.
        """
        from jax_moseq.models import arhmm, robust_arhmm

        return robust_arhmm if self.robust else arhmm

    # Everything derived from the parameters is rebuilt on demand, so none of it
    # belongs in a pickle. Excluding it here rather than only in ``__deepcopy__``
    # means a model saved directly -- a checkpoint, say -- cannot carry a stale
    # or duplicated copy either, whatever was read off it beforehand.
    _DERIVED = ("_aBl_cache", "_expected_states", "_obs_distns")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._DERIVED}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clear_derived()

    def __deepcopy__(self, memo):
        """Copy the fit without its training data.

        ``moseq2_model.util.copy_model`` exists to strip the training data
        before a model is saved, which it does by blanking the per-session
        arrays a pyhsmm states object holds. Nothing hangs off the states here;
        the data is one array on the model, so it is dropped here instead. The
        group assignments stay, since the parameters are indexed by them.
        """
        cls = self.__class__
        cp = cls.__new__(cls)
        memo[id(self)] = cp
        for key, value in self.__dict__.items():
            if key == "_data":
                value = {k: v for k, v in value.items() if k == "group"}
            elif key in ("_aBl_cache", "_expected_states", "_obs_distns"):
                # Derived from the parameters and rebuilt on demand; at
                # production scale aBl alone is larger than the fit.
                continue
            else:
                value = deepcopy(value, memo)
            setattr(cp, key, value)
        cp._clear_derived()
        return cp

    @property
    def model_dict(self):
        """The underlying jax-moseq model, for saving and checkpointing."""
        return self._model


# The four concrete classes carry the names the pyhsmm-backed classes had.
# ``get_parameters_from_model`` records ``model.__class__.__name__`` as
# ``model_parameters["model_class"]``, and moseq2-viz writes that straight into
# the crowd-movie ``info.yaml`` (``moseq2_viz/io/video.py:92``), so the name is
# user-visible and must not drift when the backend changes underneath.


class CorrectedFastARHMM(JaxARHMM):
    """Gaussian observations, one transition matrix."""


class CorrectedFastARHMMSeparateTrans(JaxARHMM):
    """Gaussian observations, one transition matrix per group."""


class CorrectedRobustARHMM(JaxARHMM):
    """Student-t observations, one transition matrix."""


class CorrectedRobustARHMMSeparateTrans(JaxARHMM):
    """Student-t observations, one transition matrix per group."""


def build(robust, separate_trans, **kwargs):
    """Construct the class matching a ``robust`` / ``separate_trans`` pair."""
    cls = {
        (False, False): CorrectedFastARHMM,
        (False, True): CorrectedFastARHMMSeparateTrans,
        (True, False): CorrectedRobustARHMM,
        (True, True): CorrectedRobustARHMMSeparateTrans,
    }[(bool(robust), bool(separate_trans))]
    return cls(robust=robust, **kwargs)


def _stack_sessions(data_dict, nlags):
    """Pad sessions to one rectangular array and mark the padding invalid.

    jax-moseq requires ``(N, T, latent_dim)``; moseq2 supplies per-session
    arrays whose lengths may differ. A frame is valid when it and all ``nlags``
    lags are finite, which is the rule moseq2-model applies implicitly by
    dropping any strided row containing a NaN.

    Returns
    -------
    x : np.ndarray of shape (N, T_max, latent_dim), NaNs zero-filled
    frame_mask : np.ndarray of shape (N, T_max), bool
    strict : np.ndarray of shape (N, T_max - nlags), bool
        Row validity, index-aligned with the sampled state sequences.
    """
    names = list(data_dict)
    t_max = max(np.asarray(data_dict[k]).shape[0] for k in names)
    latent_dim = np.asarray(data_dict[names[0]]).shape[1]

    x = np.zeros((len(names), t_max, latent_dim))
    frame_mask = np.zeros((len(names), t_max), dtype=bool)
    strict = np.zeros((len(names), t_max - nlags), dtype=bool)

    for i, name in enumerate(names):
        arr = np.asarray(data_dict[name], dtype=np.float64)
        valid = ~np.isnan(arr).any(axis=1)
        n = arr.shape[0]
        x[i, :n] = np.where(valid[:, na], arr, 0.0)
        frame_mask[i, :n] = valid

        rows = valid[nlags:].copy()
        for lag in range(1, nlags + 1):
            rows &= valid[nlags - lag : n - lag]
        strict[i, : rows.size] = rows

    frame_mask[:, nlags:] = strict
    return x, frame_mask, strict

from moseq2_pca.util import get_changepoints
from autoregressive.distributions import AutoRegression
from autoregressive.models import FastARWeakLimitStickyHDPHMM, ARWeakLimitStickyHDPHMM, ARWeakLimitStickyHDPHMMSeparateTrans
from pybasicbayes.models import Mixture
from pyhsmm.models import WeakLimitHDPHMM, WeakLimitStickyHDPHMM
import pyhsmm
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import ruamel.yaml as yaml

def compute_changepoints(datas):
    cps = []
    for k, v in datas.items():
        with np.errstate(invalid="warn"):
            _cps = get_changepoints(v.values.T, sigma=5, peak_height=1e-3, timestamps=v.index.values)[0].squeeze()
        if _cps.size > 1:
            cps += list(np.diff(_cps) / 30.)
    cps = np.array(cps)
    
    return cps

def get_syllable_slices(labels, max_syllable):#, total):
    slices, slices_idx, tmp, tmp_idx = [], [], [], []
    
    for i, l in enumerate(labels):
        if len(tmp) == 0:
            tmp.append(l)
            tmp_idx.append(i)
        elif len(tmp) > 0 and tmp[-1] == l:
            tmp.append(l)
            tmp_idx.append(i)
        elif len(tmp) > 0 and tmp[-1] != l:
            slices.append(tmp)
            slices_idx.append(tmp_idx)
            tmp_idx = [i]
            tmp = [l]
    
    syllables = range(0, max_syllable)
    dct = {}
    for syll in syllables:
        dct[syll] = {'slices': [], 'ts': []}

    for j in range(len(slices)):
        compnum = slices[j][0]
        if compnum in list(dct.keys()):
            dct[compnum]['slices'].append(slices[j])
            dct[compnum]['ts'].append(slices_idx[j])
    
    return dct

def parse_modeling_results(csvs, model_results):
    dct = {}
    dct["model"], dct["labels"], dct["ll"], dct["heldout_ll"], dct["train_idx"] = model_results
    dct["filenames"] = ([csvs[_] for _ in dct["train_idx"][0]], [csvs[_] for _ in dct["train_idx"][1]])

    return dct

def get_model(model_type, kappa, alpha, gamma, init_state_concentration, obs_dist, separate_trans=False, use_fast=False):
    # add arhmm
    if model_type.lower() == "hmm":
        model = WeakLimitHDPHMM(alpha=alpha,
                                gamma=gamma,
                                init_state_concentration=init_state_concentration,
                                obs_distns=obs_dist)
    elif model_type.lower() == "mixture":
        model = Mixture(alpha_0=alpha,
                        components=obs_dist)
    elif model_type.lower() == "stickyhmm":
        model = WeakLimitStickyHDPHMM(alpha=alpha,
                                      gamma=gamma,
                                      kappa=kappa,
                                      init_state_concentration=init_state_concentration,
                                      obs_distns=obs_dist)
    elif model_type.lower() == "arhmm" and use_fast:
        model = FastARWeakLimitStickyHDPHMM(alpha=alpha,
                                            gamma=gamma,
                                            kappa=kappa,
                                            init_state_distn="uniform",
                                            obs_distns=obs_dist)
    elif model_type.lower() == "arhmm":
        if not separate_trans:
            model = ARWeakLimitStickyHDPHMM(alpha=alpha,
                                            gamma=gamma,
                                            kappa=kappa,
                                            init_state_distn="uniform",
                                            obs_distns=obs_dist)
        else:
            model = ARWeakLimitStickyHDPHMMSeparateTrans(alpha=alpha,
                                            gamma=gamma,
                                            kappa=kappa,
                                            init_state_distn="uniform",
                                            obs_distns=obs_dist)

    return model

def get_empirical_ar_params(train_datas, params):
    obs_dim = train_datas[0].shape[1]
    # Initialize the observation parameters
    obs_params = dict(nu_0=params["nu_0"],
                      S_0=params['S_0'],
                      M_0=params['M_0'],
                      K_0=params['K_0'],
                      affine=params['affine'])

    # Fit an AR model to the entire dataset
    obs_distn = AutoRegression(**obs_params)
    obs_distn.max_likelihood(train_datas)

    # Use the inferred noise covariance as the prior mean
    # E_{IW}[S] = S_0 / (nu_0 - datadimension - 1)
    obs_params["S_0"] = obs_distn.sigma * (params["nu_0"] - obs_dim - 1)

    # make sure to include nu if we're using the tAR model
    if 'nu' in params.keys():
        obs_params['nu'] = params['nu']

    return obs_params

# this massive function handles the model training
def model_train_pbb(train_datas,  # list of numpy arrays nsamples x ndims
                    indexfile,
                    alpha=5.,  # arhmm hyper
                    gamma=120.,  # arhmm hyper
                    init_state_concentration=6.,  # hmm hyper (not ar)
                    max_states=50,  # max number of states to consider
                    kappa=None,  # stickiness (how long is each state?)
                    random_seed=40,
                    test_size=0,  # proportion of data to hold out for likelihood calculation
                    s0_scale=.01,  # usually don't need to change (ar observation hyper, already calibrated)
                    k0_scale=1e-4,  # usually don't need to change (ar observation hyper, already calibrated)
                    nlags=3,  # number of lags in ar process (3 is typical)
                    k0=0.25,  #
                    num_procs=None,  # number of processes (only use if use_fast is False)
                    model_type="arhmm",  # type of model
                    iters=100,  # number of resampling iterations
                    affine=True,  # use an affine term
                    empirical_bayes=False,  # use empirical_bayes estimate for AR observations
                    use_fast=False,  # drop down to C code for AR resampling (NOT STABLE)
                    return_model=True,  # returns the model object, useful for debugging
                    separate_trans=False, # model different groups of data
                    obs_hypparams={},
                    **kwargs):
    
    keys = list(train_datas.keys())
    data_list = [_.values for _ in train_datas.values()]
    print(keys)
    
    with open(indexfile, 'r') as f:
        index_data = yaml.safe_load(f)
    f.close()
    
    groups = [f['group'] for f in index_data['files']]
        
    
    if kappa is None:
        kappa = np.sum([len(_) for _ in data_list])
        print('Setting kappa to {:f}'.format(kappa))

    if num_procs > len(data_list):
        num_procs = len(data_list)
        print("Setting num procs <= number of session: {:d}".format(len(data_list)))

    obs_dim = data_list[0].shape[1]

    # different hypers for AR model...
    if "ar" in model_type:
        default_obs_hypparams = {
            'nu_0': obs_dim + 2,
            'S_0': s0_scale * np.eye(obs_dim),
            'M_0': np.hstack((np.eye(obs_dim),
                              np.zeros((obs_dim, obs_dim * (nlags - 1))),
                              np.zeros((obs_dim, int(affine))))),
            'affine': affine,
            'K_0': k0_scale * np.eye(obs_dim * nlags + affine)
        }
    else:
        default_obs_hypparams = {'mu_0': np.zeros(obs_dim),
                                 'sigma_0': s0_scale * np.eye(obs_dim),
                                 'kappa_0': k0,
                                 'nu_0': obs_dim + 2}
    
    obs_hypparams = {**default_obs_hypparams, **obs_hypparams}

    if "ar" in model_type:
        if empirical_bayes:
            print('Using empirical Bayes estimation of S_0')
            obs_hypparams = get_empirical_ar_params(data_list, obs_hypparams)
        obs_distns = [AutoRegression(**obs_hypparams) for _ in range(max_states)]
    else:
        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for _ in range(max_states)]

    model = get_model(model_type, kappa, alpha, gamma, init_state_concentration, obs_distns, separate_trans, use_fast)

    if test_size == 0:
        train_idx = list(range(len(data_list)))
        test_idx = []
    else:
        train_idx, test_idx = train_test_split(range(len(data_list)),
                                               random_state=random_seed,
                                               test_size=test_size)

    train_data = [data_list[idx] for idx in train_idx]
    test_data = [data_list[idx] for idx in test_idx]

    for i, (data_key, dat) in enumerate(train_datas.items()):
        if separate_trans:
            print(f'Group ID: {groups[i]}')
            model.add_data(dat, group_id=groups[i])
        else:
            model.add_data(dat)

    ll = []

    pbar = tqdm(total=iters)
    iters_run = 0
    while(True):
        if iters_run < iters:
            model.resample_model(num_procs=num_procs)
            ll.append(model.log_likelihood())
            iters_run += 1
            pbar.update(1)
        else:
            break
    pbar.close()

    heldout_ll = model.log_likelihood(test_data)

    if hasattr(model, 'states_list'):
        labels = [_.stateseq for _ in model.states_list]
    else:
        labels = [_.z for _ in model.labels_list]

    if return_model:
        return model, labels, ll, heldout_ll, (train_idx, test_idx)
    else:
        return labels, ll, heldout_ll, (train_idx, test_idx)
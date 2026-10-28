"""
Utility functions for handling loading and saving models and their respective metadata.
"""

import os
import re
import h5py
import click
import joblib
import scipy.io
import warnings
import numpy as np
from copy import deepcopy
from cytoolz import first
from collections import OrderedDict
from moseq2_model.train.models import ARHMM
from os.path import basename, getctime, join, exists


def load_pcs(filename, var_name="features", load_groups=False, npcs=10):
    """
    Load the Principal Component Scores for modeling.

    Args:
    filename (str): path to the file that contains PC scores
    var_name (str): key for pc scores in the h5 file
    load_groups (bool): Load metadata group variable
    npcs (int): Number of PCs to load

    Returns:
    data_dict (OrderedDict): key-value pairs for keys being uuids and values being PC scores.
    metadata (OrderedDict): dictionary containing lists of index-aligned uuids and groups.
    """

    metadata = {
        "uuids": None,
        "groups": {},
    }

    if filename.endswith(".mat"):
        print("Loading data from matlab file")
        data_dict = load_data_from_matlab(filename, var_name, npcs)

        # convert the uuid list to something that will export easily...
        metadata["uuids"] = load_cell_string_from_matlab(filename, "uuids")
        if load_groups:
            metadata["groups"] = dict(
                zip(metadata["uuids"], load_cell_string_from_matlab(filename, "groups"))
            )
        else:
            metadata["groups"] = None

    elif filename.endswith((".z", ".pkl", ".p")):
        print("Loading data from pickle file")
        data_dict = joblib.load(filename)

        if not isinstance(data_dict, OrderedDict):
            data_dict = OrderedDict(data_dict)

        # Reading in PCs and associated groups
        if isinstance(first(data_dict.values()), tuple):
            print("Detected tuple")
            for k, v in data_dict.items():
                data_dict[k] = v[0][:, :npcs]
                metadata["groups"][k] = v[1]
        else:
            for k, v in data_dict.items():
                data_dict[k] = v[:, :npcs]

        metadata["uuids"] = list(data_dict)

    elif filename.endswith(".h5"):
        # Reading PCs from h5 file
        with h5py.File(filename, "r") as f:
            if var_name in f:
                print(f"Found pcs in {var_name}")
                tmp = f[var_name]

                # Reading in PCs into training dict
                if isinstance(tmp, h5py.Dataset):
                    data_dict = OrderedDict([(1, tmp[:, :npcs])])

                elif isinstance(tmp, h5py.Group):
                    # Reading in PCs
                    data_dict = OrderedDict([(k, v[:, :npcs]) for k, v in tmp.items()])
                    # Optionally loading groups if they
                    if load_groups:
                        if "groups" in f:
                            metadata["groups"] = {
                                key: f["groups"][i]
                                for i, key in enumerate(data_dict)
                                if key in f["metadata"]
                            }
                        else:
                            warnings.warn(
                                "groups key not found in h5 file, assigning each session to unique group if no moseq2-index.yaml"
                            )
                            metadata["groups"] = {
                                key: i for i, key in enumerate(data_dict)
                            }
                    else:
                        warnings.warn(
                            "groups not loaded from h5 file. Assigning each session to unique group if no moseq2-index.yaml"
                        )
                        metadata["groups"] = {key: i for i, key in enumerate(data_dict)}
                else:
                    raise IOError("Could not load data from h5 file")
            else:
                raise IOError(f"Could not find dataset name {var_name} in {filename}")

            # if all the h5 data keys are uuids, use them to store uuid information
            if all(map(is_uuid, data_dict)):
                metadata["uuids"] = list(data_dict)
            elif "metadata" in f:
                metadata["uuids"] = list(f["metadata"])
    else:
        raise ValueError("Did not understand filetype")

    return data_dict, metadata


def is_uuid(string):
    """
    Check to see if string is a uuid.

    Args:
    string (str): string containing a session uuid in the index file

    Returns:
    (bool): boolean to indicate if a string is a uuid.
    """
    regex = re.compile(
        r"^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z",
        re.I,
    )
    match = regex.match(string)
    return bool(match)


def get_current_model(use_checkpoint, all_checkpoints, train_data, model_parameters):
    """
    Load the latest model checkpoint of use_checkppoint parameter is True, otherwise instantiate a new model.

    Args:
    use_checkpoint (bool): flag that indicates whether to load a checkpointed model
    all_checkpoints (list): list of all found checkpoint paths
    train_data (OrderedDict): dictionary of uuid-PC score key-value pairs
    model_parameters (dict): dictionary of required modeling hyperparameters.

    Returns:
    arhmm (ARHMM): instantiated model object including loaded data
    itr (int): starting iteration number for the model to begin training from.
    """

    # Check for available previous modeling checkpoints
    itr = 0
    if use_checkpoint and len(all_checkpoints) > 0:
        # Get latest checkpoint (with respect to save date)
        latest_checkpoint = max(all_checkpoints, key=getctime)
        click.echo(f"Loading Checkpoint: {basename(latest_checkpoint)}")
        try:
            checkpoint = load_arhmm_checkpoint(latest_checkpoint, train_data)
            # Get model object
            arhmm = checkpoint.pop("model")
            itr = checkpoint.pop("iter")
            click.echo(f"On iteration {itr}")
        except (FileNotFoundError, ValueError):
            click.echo("Loading original checkpoint failed, creating new ARHMM")
            arhmm = ARHMM(data_dict=train_data, **model_parameters)
    else:
        if use_checkpoint:
            print("No checkpoints found.", end=" ")
        click.echo("Creating new ARHMM")
        arhmm = ARHMM(data_dict=train_data, **model_parameters)

    return arhmm, itr


def get_loglikelihoods(arhmm, data, groups, separate_trans, normalize=True):
    """
    Compute the log-likelihoods of the training sessions.

    Args:
    arhmm (ARHMM): the ARHMM model object.
    data (dict): dict object with UUID keys containing the PCS used for training.
    groups (list): list of assigned groups for all corresponding session uuids.
    separate_trans (bool): flag to compute separate log-likelihoods for each modeled group.
    normalize (bool): if set to True this function will normalize by frame counts in each session

    Returns:
    ll (list): list of log-likelihoods for the trained model
    """

    if separate_trans:
        ll = [
            arhmm.log_likelihood(v, group_id=g) for g, v in zip(groups, data.values())
        ]
    else:
        ll = [arhmm.log_likelihood(v) for v in data.values()]
    if normalize:
        ll = [l / len(v) for l, v in zip(ll, data.values())]

    return ll


def get_session_groupings(data_metadata, train_list, hold_out_list):
    """
    Create a list or tuple of assigned groups for training and (optionally) held out data.

    Args:
    data_metadata (dict): dict containing session group information
    all_keys (list): list of all corresponding included session uuids
    hold_out_list (list): list of held-out uuids

    Returns:
    groupings (tuple): 2-tuple containing lists of train groups
    and held-out groups (if held_out_list exists)
    """

    # Get held out groups
    hold_g = [data_metadata["groups"][k] for k in hold_out_list]
    train_g = [data_metadata["groups"][k] for k in train_list]

    # Ensure training groups were found before setting grouping
    if len(train_g) != 0:
        return train_g, hold_g
    return None


def save_dict(filename, obj_to_save=None):
    """
    Save dictionary to file.

    Args:
    filename (str): path to file where dict is being saved.
    obj_to_save (dict): dict to save.
    """

    # Parsing given file extension and saving model accordingly
    if filename.endswith(".mat"):
        print("Saving MAT file", filename)
        scipy.io.savemat(filename, mdict=obj_to_save)
    elif filename.endswith(".z"):
        print("Saving compressed pickle", filename)
        joblib.dump(obj_to_save, filename, compress=("zlib", 4))
    elif filename.endswith((".pkl", ".p")):
        print("Saving pickle", filename)
        joblib.dump(obj_to_save, filename, compress=0)
    elif filename.endswith(".h5"):
        print("Saving h5 file", filename)
        with h5py.File(filename, "w") as f:
            dict_to_h5(f, obj_to_save)
    else:
        raise ValueError("Did not understand filetype")


def load_dict(filename):
    """
    Load dictionary from file.

    Args:
        filename (str): path to file where dict is saved

    Returns:
        obj (dict): loaded dictionary
    """
    if filename.endswith(".h5"):
        return h5_to_dict(filename)
    elif filename.endswith(("pkl", "p", "z")):
        return joblib.load(filename)
    else:
        raise ValueError(f"Does not support filetype {os.path.splitext(filename)[1]}")


def dict_to_h5(h5file, export_dict, path="/"):
    """
    Recursively save dicts to h5 file groups.

    Args:
    h5file (h5py.File): opened h5py File object.
    export_dict (dict): dictionary to save
    path (str): path within h5 to save to.

    Returns:
    """
    # https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    for key, item in export_dict.items():
        # Parse key and value types, and load them accordingly
        if isinstance(key, (tuple, int)):
            key = str(key)
        if isinstance(item, str):
            item = item.encode("utf8")

        # Write dict item to h5 based on its data-type
        if isinstance(item, np.ndarray) and item.dtype == object:
            dt = h5py.special_dtype(vlen=np.array(item.flat[0]).dtype)
            h5file.create_dataset(path + key, item.shape, dtype=dt, compression="gzip")
            for tup, _ in np.ndenumerate(item):
                if item[tup] is not None:
                    h5file[path + key][tup] = np.array(item[tup]).ravel()
        elif isinstance(item, (np.ndarray, list)):
            h5file.create_dataset(path + key, data=item, compression="gzip")
        elif isinstance(item, (int, float, str, bytes)):
            h5file.create_dataset(path + key, data=item)
        elif isinstance(item, dict):
            dict_to_h5(h5file, item, path + key + "/")
        else:
            raise ValueError(f"Cannot save {type(item)} type")


def load_arhmm_checkpoint(filename: str, train_data: dict) -> dict:
    """
    Load an arhmm checkpoint and add data into the arhmm model checkpoint.

    Args:
    filename (str): path that specifies the checkpoint.
    train_data (OrderedDict): an OrderedDict that contains the training data

    Returns:
    mdl_dict (dict): a dict containing the model with reloaded data, and associated training data
    """

    # The checkpoint was written through copy_model, which strips the training
    # data, so the model can report its parameters but cannot resample until
    # the sessions are put back.
    mdl_dict = joblib.load(filename)
    mdl_dict["model"].attach_data(train_data)

    return mdl_dict


def save_arhmm_checkpoint(filename: str, arhmm: dict):
    """
    Save an arhmm checkpoint.

    Args:
    filename (str): path that specifies the checkpoint
    arhmm (dict): a dictionary containing the arhmm object, training iteration number, log-likelihoods of each training step, and labels for each step.
    """

    # Getting model object
    mdl = arhmm.pop("model")
    arhmm["model"] = copy_model(mdl)

    # Save model
    print(f"Saving Checkpoint {filename}")
    joblib.dump(arhmm, filename, compress=("zlib", 5))


def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    """
    Load the contents of an h5 file at a specified path into a dictionary.

    Args:
    filename (h5py.File): opened h5 file.
    path (str): path within the h5 file to load data from.

    Returns:
    (dict): dict containing all of the h5 file contents.
    """

    ans = {}
    if isinstance(file[path], h5py._hl.dataset.Dataset):
        # only use the final path key to add to `ans`
        ans[path.split("/")[-1]] = file[path][()]
    else:
        # Reading in h5 value into dict key-value pair
        for key, item in file[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = _load_h5_to_dict(file, "/".join([path, key]))
    return ans


def h5_to_dict(h5file, path: str = "/") -> dict:
    """
    Load h5 data to dictionary from a user specified path.

    Args:
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    out (dict): a dict with h5 file contents with the same path structure
    """

    # Load h5 file according to whether it is separated by Groups
    if isinstance(h5file, str):
        with h5py.File(h5file, "r") as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, (h5py.File, h5py.Group)):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception("file input not understood - need h5 file path or file object")
    return out


def load_data_from_matlab(filename, var_name="features", npcs=10):
    """
    Load PC Scores from a specified variable column in a MATLAB file.

    Args:
    filename (str): path to MATLAB (.mat) file
    var_name (str): variable to load
    npcs (int): number of PCs to load.

    Returns:
    data_dict (OrderedDict): loaded dictionary of uuid and PC-score pairings.
    """

    data_dict = OrderedDict()

    with h5py.File(filename, "r") as f:
        # Loading PCs scores into training data dict
        if var_name in f:
            score_tmp = f[var_name]
            for i in range(len(score_tmp)):
                tmp = f[score_tmp[i][0]]
                score_to_add = tmp[()]
                data_dict[i] = score_to_add[:npcs, :].T

    return data_dict


def load_cell_string_from_matlab(filename, var_name="uuids"):
    """
    Load cell strings from MATLAB file.

    Args:
    filename (str): path to .mat file
    var_name (str): variable name to read

    Returns:
    return_list (list): list of selected loaded variables
    """

    return_list = []
    with h5py.File(filename, "r") as f:

        if var_name in f:
            tmp = f[var_name]

            # change unichr to chr for python 3
            for i in range(len(tmp)):
                tmp2 = f[tmp[i][0]]
                uni_list = ["".join(chr(c.item())) for c in tmp2]
                return_list.append("".join(uni_list))

    return return_list


# per Scott's suggestion
def copy_model(model_obj):
    """
    Return a deep copy of the ARHMM that doesn't contain the training data.

    Args:
    model_obj (ARHMM): model to copy.

    Returns:
    cp (ARHMM): copy of the model
    """

    tmp = []

    # make a deep copy of the data-less version
    for s in model_obj.states_list:
        tmp.append(s.data)
        s.data = None

    cp = deepcopy(model_obj)

    # now put the data back in the model for training

    for s, t in zip(model_obj.states_list, tmp):
        s.data = t

    return cp


def get_parameters_from_model(model):
    """
    Get parameter dictionary from model.

    Args:
    model (ARHMM): model to get parameters from.

    Returns:
    parameters (dict): dictionary containing all modeling parameters
    """

    init_obs_dist = model.init_emission_distn.hypparams

    # Loading transition graph(s)
    if hasattr(model, "trans_distns"):
        trans_dist = model.trans_distns[0]
    else:
        trans_dist = model.trans_distn

    ls_obj = dir(model.obs_distns[0])

    # Packing object parameters into a single dict
    parameters = {
        "kappa": trans_dist.kappa,
        "gamma": trans_dist.gamma,
        "alpha": trans_dist.alpha,
        "nu": np.nan,
        "max_states": trans_dist.N,
        "nu_0": init_obs_dist["nu_0"],
        "sigma_0": init_obs_dist["sigma_0"],
        "kappa_0": init_obs_dist["kappa_0"],
        "nlags": model.nlags,
        "mu_0": init_obs_dist["mu_0"],
        "model_class": model.__class__.__name__,
        "ar_mat": [obs.A for obs in model.obs_distns],
        "sig": [obs.sigma for obs in model.obs_distns],
    }

    if "nu" in ls_obj:
        parameters["nu"] = [obs.nu for obs in model.obs_distns]

    return parameters


def count_frames(data_dict=None, input_file=None, var_name="scores"):
    """
    Count the total number of frames loaded from the PC scores file.

    Args:
    data_dict (OrderedDict): Loaded PC scores OrderedDict object.
    input_file (str): Path to PC Scores file to load data_dict if not already data_dict is None
    var_name (str): Path within PCA h5 file to load scores from.

    Returns:
    total_frames (int): total number of counted frames.
    """

    if data_dict is None and input_file is not None:
        data_dict, _ = load_pcs(
            filename=input_file, var_name=var_name, load_groups=True
        )

    total_frames = 0
    for v in data_dict.values():
        idx = (~np.isnan(v)).all(axis=1)
        total_frames += np.sum(idx)
    print("The total number of frames:", total_frames)
    return total_frames


def get_parameter_strings(config_data):
    """
    Create the CLI learn-model command using the given config_data dict contents to run the modeling step.

    Args:
    config_data (dict): Configuration parameters dict.

    Returns:
    parameters (str): String containing CLI command parameter flags.
    prefix (str): Prefix string for the learn-model command (Slurm only).
    """

    parameters = f' --npcs {config_data["npcs"]} --num-iter {config_data["num_iter"]} '

    if isinstance(config_data["index"], str):
        if exists(config_data["index"]):
            parameters += f'-i {config_data["index"]} '

    if config_data["separate_trans"]:
        parameters += "--separate-trans "

    if config_data["robust"]:
        parameters += "--robust "

    if config_data["e_step"]:
        parameters += "--e-step "

    if config_data["hold_out"]:
        parameters += f'--hold-out --nfolds {config_data["nfolds"]} '

    if config_data["max_states"]:
        parameters += f'--max-states {config_data["max_states"]} '
    if config_data["ncpus"] > 0:
        parameters += f'--ncpus {config_data["ncpus"]} '

    # Handle possible Slurm batch functionality
    prefix = ""
    if config_data["cluster_type"] == "slurm":
        prefix = f'sbatch -c {config_data["ncpus"] if config_data["ncpus"] > 0 else 8} --mem={config_data["memory"]} '
        prefix += f'-p {config_data["partition"]} -t {config_data["wall_time"]} --wrap "{config_data["prefix"]}'

    return parameters, prefix


def create_command_strings(
    input_file, output_dir, config_data, kappas, model_name_format="model-{:03d}-{}.p"
):
    """
    Create the CLI learn-model command strings with parameter flags based on the contents of the configuration dict.

    Args:
    input_file (str): Path to PC Scores
    output_dir (str): Path to directory to save models in.
    config_data (dict): Configuration parameters dict.
    kappas (list): List of kappa values for model training commands.
    model_name_format (str): Filename string format string.

    Returns:
    command_string (str): CLI learn-model command strings with the requested parameters separated by newline characters
    """

    # Get base command and parameter flags
    base_command = f"moseq2-model learn-model {input_file} "
    parameters, prefix = get_parameter_strings(config_data)

    commands = []
    for i, k in enumerate(kappas):
        # Create CLI command
        cmd = (
            base_command
            + join(output_dir, model_name_format.format(i, k))
            + parameters
            + f"--kappa {k}"
        )

        # Add possible batch fitting prefix string
        if config_data["cluster_type"] == "slurm":
            cmd = prefix + cmd + '"'
        commands.append(cmd)

    # Create and return the command string
    command_string = "\n".join(commands)
    command_string = "set -e\n" + command_string
    return command_string


def get_scan_range_kappas(data_dict, config_data):
    """
    Get the kappa values to train models on based on the user's selected scanning scale range.
    Default values will be selected if min/max_kappa are None.

    An example: scan_scale = 'log'; nframes = 1800; min_kappa = 10e3; max_kappa = 10e5; n_models = 10;
    >>> kappas = [1000, 1668, 2782, 4641, 7742, 12915, 21544, 35938, 59948, 100000]

    Another Exmaple:
    nframes = 1800
    'scan_scale': 'linear',
    'min_kappa': None,
    'max_kappa': None,
    'n_models': 10
    min(kappas) == 18
    max(kappas) == 18000000
    >>> kappas == [18, 20016, 40014, 60012, 80010, 100008, 120006, 140004, 160002, 180000]

    Args:
    data_dict (OrderedDict): Loaded PCA score dictionary.
    config_data (dict): Configuration parameters dict.

    Returns:
    kappas (list): list of ints corresponding to the kappa value for each model.
    """

    nframes = count_frames(data_dict)

    if config_data.get("scan_scale", "log") == "log":
        # Get log scan range
        factor = int(np.log10(nframes))
        if config_data["min_kappa"] is None:
            min_factor = factor - 2  # Set default value
        else:
            min_factor = np.log10(config_data["min_kappa"])

        if config_data["max_kappa"] is None:
            max_factor = factor + 2  # Set default value
        else:
            max_factor = np.log10(config_data["max_kappa"])

        kappas = np.round(
            np.logspace(min_factor, max_factor, config_data["n_models"])
        ).astype("int")
        config_data["min_kappa"] = kappas[0]
        config_data["max_kappa"] = kappas[-1]

    elif config_data["scan_scale"] == "linear":
        # Get linear scan range
        # Handle either of the missing parameters
        if config_data["min_kappa"] is None:
            # Choosing a minimum kappa value (AKA value to begin the scan from)
            # less than the counted number of frames
            config_data["min_kappa"] = min(
                nframes, nframes / 1e2
            )  # default initial kappa value
        if config_data["max_kappa"] is None:
            # If no max is specified, max kappa will be 100x the number of frames.
            config_data["max_kappa"] = max(
                nframes, nframes * 1e2
            )  # default initial kappa values

        kappas = np.linspace(
            config_data["min_kappa"], config_data["max_kappa"], config_data["n_models"]
        ).astype("int")

    return kappas

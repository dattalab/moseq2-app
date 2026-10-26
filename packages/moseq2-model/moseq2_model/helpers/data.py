"""
Helper functions for reading data from index files, and preparing metadata prior to training.
"""

import click
import random
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from cytoolz import pluck, curried
from collections import OrderedDict
from os.path import join, exists, dirname
from moseq2_model.util import count_frames
from moseq2_model.train.models import flush_print
from moseq2_model.train.util import whiten_all, whiten_each
from moseq2_model._yaml import read_yaml


def process_indexfile(index, data_metadata, default_group="n/a", select_groups=False):
    """
    Read index file (if applicable) and return dictionaries containing metadata in the index file.

    Args:
    index (str or None): path to index file.
    data_metadata (dict): loaded metadata containing uuid and group information.
    default_group (str): default group name to supply to data without assigned group labels
    select_groups (bool): when True, print metadata describing group selection

    Returns:
    index_data (dict): loaded index file.
    data_metadata (dict): updated metadata dictionary.
    """

    # if we have an index file, strip out the groups, match to the scores uuids
    if index is not None and exists(index):
        with open(index, "r") as f:
            # reading in array of files
            index_data = read_yaml(f)
            yml_metadata = index_data["files"]

        # reading corresponding groups and uuids
        uuid_map = dict(pluck(["uuid", "group"], yml_metadata))

        # Setting model metadata group array
        data_metadata["groups"] = uuid_map

        # Optionally display metadata to select groups to model
        if select_groups:
            get_subject_name = curried.get_in(
                ["metadata", "SubjectName"], default="default"
            )
            get_session_name = curried.get_in(
                ["metadata", "SessionName"], default="default"
            )
            for i, data in enumerate(index_data["files"], start=1):
                print(
                    f"[{i}]",
                    "Session Name:",
                    get_session_name(data),
                    "; Subject Name:",
                    get_subject_name(data),
                    "; Group:",
                    data["group"],
                    "; Key:",
                    data["uuid"],
                )
    else:
        index_data = None

    return index_data, data_metadata


def select_data_to_model(index_data, data_dict, data_metadata, select_groups=False):
    """
    Select the data to model.

    Args:
    index_data (dict): loaded dictionary from index file
    data_dict (dict): dictionary containing PC scores for all sessions
    data_metadata (dict): dictionary containing metadata associated with the recording sessions
    select_groups (bool): flag to solicit user input on which groups to select for modeling

    Returns:
    data_dict (dict): dictionary to model containing data from the selected session uuids
    data_metadata (dict): updated metadata containing the selected uuids and groups
    """

    # If no input is given, load all the uuids and groups
    use_keys, use_groups = zip(*pluck(["uuid", "group"], index_data["files"]))

    if select_groups:
        # Prompt user to select groups to include in training set
        print("Select from the following groups:", list(np.unique(use_groups)))
        groups_to_train = input(
            "Input comma/space-separated names of the groups to model. "
            "Empty string to model all the sessions/groups in the index file."
        )
        if "," in groups_to_train:
            # Parse multiple inputted groups if comma is found
            sel_groups = [g.strip() for g in groups_to_train.split(",")]
            use_keys = [
                f["uuid"] for f in index_data["files"] if f["group"] in sel_groups
            ]
            use_groups = [
                f["group"] for f in index_data["files"] if f["uuid"] in use_keys
            ]
        elif len(groups_to_train) > 0:
            # Parse multiple groups in case input is delimited by a space
            sel_groups = [g for g in groups_to_train.split(" ")]
            use_keys = [
                f["uuid"] for f in index_data["files"] if f["group"] in sel_groups
            ]
            use_groups = [
                f["group"] for f in index_data["files"] if f["uuid"] in use_keys
            ]

    use_groups = dict(zip(use_keys, use_groups))

    data_dict = OrderedDict((k, data_dict[k]) for k in use_keys)
    data_metadata["uuids"] = use_keys
    data_metadata["groups"] = use_groups

    return data_dict, data_metadata


def prepare_model_metadata(data_dict, data_metadata, config_data):
    """
    Set model training metadata parameters, whiten data, split data and return list of heldout keys if applicable, and update all dictionaries.

    Args:
    data_dict (OrderedDict): loaded data dictionary.
    data_metadata (OrderedDict): loaded metadata dictionary.
    config_data (dict): dictionary containing all modeling parameters.

    Returns:
    data_dict (OrderedDict): optionally whitened and updated data dictionary.
    model_parameters (dict): model parameters used to initialize the ARHMM
    train_list (list): list of session uuids to include for training.
    hold_out_list (list): list of session uuids to hold out (if hold_out == True)
    """

    if config_data["kappa"] is None:
        # Count total number of frames, then set it as kappa
        total_frames = count_frames(data_dict)
        flush_print(f"Setting kappa to the number of frames: {total_frames}")
        config_data["kappa"] = total_frames

    # Optionally hold out sessions for testing
    if config_data["hold_out"] and len(data_dict) >= config_data["nfolds"]:
        click.echo(f"Will hold out 1 fold of {config_data['nfolds']}")

        if config_data["hold_out_seed"] >= 0:
            # Select repeatable random sessions to hold out
            click.echo(f"Settings random seed to {config_data['hold_out_seed']}")
            rnd = random.Random(config_data["hold_out_seed"])
        else:
            # Holding out sessions randomly
            warnings.warn(
                "Random seed not set, will choose a different test set each time this is run..."
            )
            rnd = random
        # sample all uuids, split into nfolds
        splits = np.array_split(
            rnd.sample(list(data_dict), len(data_dict)), config_data["nfolds"]
        )

        # Make list of held out session uuids
        hold_out = list(splits[0])
        # Put remainder of the data in the training set
        train = list(np.concatenate(splits[1:]))
        click.echo("Holding out " + str(hold_out))
        click.echo("Training on " + str(train))
    else:
        click.echo("Training model on all sessions")
        # Set train list to all the uuids
        config_data["hold_out"] = False
        hold_out = []
        train = list(data_dict)

    if config_data["ncpus"] > len(train):
        # Setting number of allocated cpus equal to number of training sessions
        config_data["ncpus"] = len(train)
        warnings.warn(
            f"Setting ncpus to {len(train)}. ncpus must be <= number of training sessions in dataset"
        )

    # Pack all the modeling parameters into a single dict
    model_parameters = {
        "gamma": config_data["gamma"],
        "alpha": config_data["alpha"],
        "kappa": config_data["kappa"],
        "nlags": config_data["nlags"],
        "robust": config_data["robust"],
        "max_states": config_data["max_states"],
        "separate_trans": config_data["separate_trans"],
    }

    # Adding groups to modeling parameters to compute separate transition graphs
    if config_data["separate_trans"]:
        model_parameters["groups"] = {k: data_metadata["groups"][k] for k in train}

    # Whiten the data
    whitening_parameters = None
    if config_data["whiten"][0].lower() == "a":
        click.echo("Whitening the training data using the whiten_all function")
        # in this case, whitening_parameters is a single tuple
        data_dict, whitening_parameters = whiten_all(data_dict)
    elif config_data["whiten"][0].lower() == "e":
        click.echo("Whitening the training data using the whiten_each function")
        # in this case, whitening_parameters is a dictionary of parameters
        data_dict, whitening_parameters = whiten_each(data_dict)
    else:
        click.echo("Not whitening the data")

    # Applying Additive White Gaussian Noise
    if config_data["noise_level"] > 0:
        click.echo(f'Using {config_data["noise_level"]} STD AWGN.')
        for k, v in data_dict.items():
            data_dict[k] = v + np.random.randn(*v.shape) * config_data["noise_level"]

    return data_dict, model_parameters, train, hold_out, whitening_parameters


def get_heldout_data_splits(data_dict, train_list, hold_out_list):
    """
    Split data by session UUIDs into training and held out datasets.

    Args:
    data_dict (OrderedDict): dictionary of all PC scores included in the model
    train_list (list): list of keys included in the training data
    hold_out_list (list): list of keys included in the held out data

    Returns:
    train_data (OrderedDict): dictionary of uuid to PC score key-value pairs for uuids in train_list
    test_data (OrderedDict): dictionary of uuids to PC score key-value pairs for uuids in hold_out_list.
    """

    # Getting OrderedDicts of the training, and testing/held-out data
    train_data = OrderedDict((i, data_dict[i]) for i in train_list)
    test_data = OrderedDict((i, data_dict[i]) for i in hold_out_list)

    return train_data, test_data


def get_training_data_splits(split_frac, data_dict):
    """
    Split the data into a training and held out dataset by splitting each session by some fraction `percent_split`.

    Args:
    split_frac (float): fraction to split each session into training data, leaving the rest as validation data.
    data_dict (OrderedDict): dict of uuid-PC Score key-value pairs for all data included in the model.

    Returns:
    training_data (OrderedDict): the split percentage of the training data.
    validation_data (OrderedDict): the split percentage of the validation data
    """

    if not 0 < split_frac < 1:
        raise ValueError('split_frac must be strictly between 0 and 1')

    training_data = OrderedDict()
    validation_data = OrderedDict()

    for k, v in data_dict.items():
        # Splitting data by test set
        training_X, testing_X = (
            v[: int(len(v) * split_frac)],
            v[int(len(v) * split_frac) :],
        )

        # Setting training data key-value pair
        training_data[k] = training_X

        # Setting validation data key-value pair
        validation_data[k] = testing_X

    return training_data, validation_data


def graph_modeling_loglikelihoods(config_data, iter_lls, iter_holls, model_dir):
    """
    Graph model training performance progress throughout modeling if verbose is True

    Args:
    config_data (dict): dictionary of model training parameters.
    iter_lls (list): list of training log-likelihoods for each training iteration
    iter_holls (list): list of held out log-likelihoods for each training iteration
    model_dir (str): path to the directory the model is saved in.

    Returns:
    img_path (str): path to saved graph.
    """

    ll_type = "validation"
    if config_data["hold_out"]:
        ll_type = "held_out"

    iterations = np.arange(len(iter_lls))

    plt.plot(iterations, iter_lls, label="training")
    if len(iter_holls) > 0:
        plt.plot(iterations, iter_holls, label=ll_type)

    plt.legend()
    plt.ylabel("Average Syllable Log-Likelihood")
    plt.xlabel("Iterations")

    # Saving plots
    if config_data["hold_out"]:
        img_path = join(model_dir, "train_heldout_summary.png")
        plt.title(
            "ARHMM Training Summary With " + str(config_data["nfolds"]) + " Folds"
        )
        plt.savefig(img_path, dpi=300)
    else:
        img_path = join(
            model_dir, f'train_val{config_data["percent_split"]}_summary.png'
        )
        plt.title(
            "ARHMM Training Summary With "
            + str(config_data["percent_split"])
            + "% Train-Val Split"
        )
        plt.savefig(img_path, dpi=300)

    return img_path

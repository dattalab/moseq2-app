"""
Wrapper functions for CLI and GUI.
"""

import os
import sys
import glob
import click
import numpy as np
from copy import deepcopy
from cytoolz import valmap
from moseq2_model.train.util import train_model, run_e_step, apply_model
from os.path import join, basename, realpath, dirname, splitext
from moseq2_model.util import (
    save_dict,
    load_pcs,
    get_parameters_from_model,
    copy_model,
    get_scan_range_kappas,
    create_command_strings,
    get_current_model,
    get_loglikelihoods,
    get_session_groupings,
    load_dict,
)
from moseq2_model.helpers.data import (
    process_indexfile,
    select_data_to_model,
    prepare_model_metadata,
    graph_modeling_loglikelihoods,
    get_heldout_data_splits,
    get_training_data_splits,
)


def learn_model_wrapper(input_file, dest_file, config_data):
    """
    Wrapper function to train ARHMM on PC scores.

    Args:
    input_file (str): path to pc scores file.
    dest_file (str): path to save model to.
    config_data (dict): dictionary containing the modeling parameters.

    Returns:
    None
    """

    dest_file = realpath(dest_file)
    # make sure the extension for model is correct
    assert splitext(basename(dest_file))[-1] in [
        ".mat",
        ".z",
        ".pkl",
        ".p",
        ".h5",
    ], "Incorrect model filetype"
    os.makedirs(dirname(dest_file), exist_ok=True)

    if not os.access(dirname(dest_file), os.W_OK):
        raise IOError("Output directory is not writable.")

    # Handle checkpoint parameters
    checkpoint_path = join(dirname(dest_file), "checkpoints/")
    checkpoint_freq = config_data.get("checkpoint_freq", -1)

    if checkpoint_freq < 0:
        checkpoint_freq = None
    else:
        os.makedirs(checkpoint_path, exist_ok=True)

    click.echo("Entering modeling training")

    run_parameters = deepcopy(config_data)

    # Get session PC scores and session metadata dicts
    data_dict, data_metadata = load_pcs(
        filename=input_file,
        var_name=config_data.get("var_name", "scores"),
        npcs=config_data["npcs"],
        load_groups=config_data["load_groups"],
    )

    # Parse index file and update metadata information; namely groups
    # If no group data in pca data, use group info from index file
    select_groups = config_data.get("select_groups", False)
    index_data, data_metadata = process_indexfile(
        config_data.get("index", None),
        data_metadata,
        config_data["default_group"],
        select_groups,
    )

    # Get keys to include in training set
    # TODO: select_groups not implemented
    if index_data is not None:
        data_dict, data_metadata = select_data_to_model(
            index_data, data_dict, data_metadata, select_groups
        )

    all_keys = list(data_dict)
    groups = list(data_metadata["groups"].values())

    # Get train/held out data split uuids
    data_dict, model_parameters, train_list, hold_out_list, whitening_parameters = (
        prepare_model_metadata(data_dict, data_metadata, config_data)
    )

    # Pack data dicts corresponding to uuids in train_list and hold_out_list
    if config_data["hold_out"]:
        train_data, test_data = get_heldout_data_splits(
            data_dict, train_list, hold_out_list
        )
    elif config_data["percent_split"] > 0:
        # If not holding out sessions, split the data into a validation set with the percent_split option
        train_data, test_data = get_training_data_splits(
            config_data["percent_split"] / 100, data_dict
        )
    else:
        # use all the data if percent split is 0 or lower
        train_data = data_dict
        test_data = None

    # Get all saved checkpoints
    checkpoint_file = basename(dest_file).replace(".p", "")
    all_checkpoints = glob.glob(join(checkpoint_path, f"{checkpoint_file}*.arhmm"))

    # Instantiate model; either anew or from previously saved checkpoint
    arhmm, itr = get_current_model(
        config_data["use_checkpoint"], all_checkpoints, train_data, model_parameters
    )

    # Pack progress bar keyword arguments
    progressbar_kwargs = {
        "total": config_data["num_iter"],
        "file": sys.stdout,
        "leave": False,
        "disable": not config_data["progressbar"],
        "initial": itr,
    }

    # Get data groupings for verbose train vs. test log-likelihood estimation and graphing
    if hold_out_list is not None and groups is not None:
        groupings = get_session_groupings(data_metadata, train_list, hold_out_list)
    else:
        groupings = None

    # Train ARHMM
    arhmm, loglikes, labels, iter_lls, iter_holls, interrupt = train_model(
        model=arhmm,
        num_iter=config_data["num_iter"],
        ncpus=config_data["ncpus"],
        checkpoint_freq=checkpoint_freq,
        checkpoint_file=join(checkpoint_path, checkpoint_file),
        start=itr,
        progress_kwargs=progressbar_kwargs,
        train_data=train_data,
        val_data=test_data,
        separate_trans=config_data["separate_trans"],
        groups=groupings,
        check_every=config_data["check_every"],
        verbose=config_data["verbose"],
    )

    click.echo("Computing likelihoods on each training dataset...")
    # Get training log-likelihoods
    train_ll = get_loglikelihoods(
        arhmm, train_data, groupings[0], config_data["separate_trans"]
    )

    heldout_ll = []
    # Get held out log-likelihoods
    if config_data["hold_out"]:
        click.echo("Computing held out likelihoods with separate transition matrix...")
        heldout_ll = get_loglikelihoods(
            arhmm, test_data, groupings[1], config_data["separate_trans"]
        )

    save_parameters = get_parameters_from_model(arhmm)

    if config_data["e_step"]:
        click.echo("Running E step...")
        expected_states = run_e_step(arhmm)

    # TODO: just compute cross-likes at the end and potentially dump the model (what else
    # would we want the model for hm?), though hard drive space is cheap, recomputing models is not...

    # Pack model data
    export_dict = {
        "loglikes": loglikes,
        "labels": labels,
        "keys": all_keys,
        "heldout_ll": heldout_ll,
        "model_parameters": save_parameters,
        "run_parameters": run_parameters,
        "metadata": data_metadata,
        "model": copy_model(arhmm) if config_data.get("save_model", True) else None,
        "hold_out_list": hold_out_list,
        "train_list": train_list,
        "train_ll": train_ll,
        "expected_states": expected_states if config_data["e_step"] else None,
        "whitening_parameters": whitening_parameters,
        "pc_score_path": os.path.abspath(input_file),
    }

    # Save model
    save_dict(filename=dest_file, obj_to_save=export_dict)

    if interrupt:
        raise KeyboardInterrupt()

    if config_data["verbose"] and len(iter_lls) > 0:
        img_path = graph_modeling_loglikelihoods(
            config_data, iter_lls, iter_holls, dirname(dest_file)
        )
        return img_path


def apply_model_wrapper(model_file, pc_file, dest_file, config_data):
    """
    Wrapper function to apply a pre-trained model to new data.

    Args:
    model_file (str): Path to pre-trained model file
    pc_file (str): Path to PC scores file
    dest_file (str): Path to save output file

    Returns:
    None
    """

    assert splitext(basename(dest_file))[-1] in [
        ".mat",
        ".z",
        ".pkl",
        ".p",
        ".h5",
    ], "Incorrect model filetype"
    os.makedirs(dirname(dest_file), exist_ok=True)

    if not os.access(dirname(dest_file), os.W_OK):
        raise IOError("Output directory is not writable.")

    # Load model
    model_data = load_dict(model_file)

    if model_data.get("whitening_parameters") is None:
        raise KeyError(
            "Whitening parameters not found in model file. Unable to apply model to new data. Please retrain the model using the latest version."
        )

    # Load PC scores
    data_dict, data_metadata = load_pcs(
        filename=pc_file,
        var_name=config_data.get("var_name", "scores"),
        npcs=model_data["run_parameters"]["npcs"],
        load_groups=config_data.get("load_groups", False),
    )

    # parse group information from index file
    index_data, data_metadata = process_indexfile(
        config_data.get("index", None),
        data_metadata,
        config_data.get("default_group", "n/a"),
        select_groups=False,
    )

    # Apply model
    syllables = apply_model(
        model_data["model"],
        model_data["whitening_parameters"],
        data_dict,
        data_metadata,
        model_data["run_parameters"]["whiten"],
    )

    # add -5 padding to the list of states
    nlags = model_data["run_parameters"].get("nlags", 3)
    syllables = valmap(lambda v: np.concatenate(([-5] * nlags, v)), syllables)

    # prepare model data dictionary to save
    # save applied model data
    applied_model_data = {}
    applied_model_data["labels"] = list(syllables.values())
    applied_model_data["keys"] = list(syllables.keys())
    applied_model_data["metadata"] = data_metadata
    applied_model_data["pc_score_path"] = os.path.abspath(pc_file)
    applied_model_data["pre_trained_model_path"] = os.path.abspath(model_file)

    # copy over pre-trained model data
    for key in ["model_parameters", "run_parameters", "model", "whitening_parameters"]:
        applied_model_data[key] = model_data[key]

    # Save output
    save_dict(filename=dest_file, obj_to_save=applied_model_data)


def kappa_scan_fit_models_wrapper(input_file, config_data, output_dir):
    """
    Wrapper function to output multiple model training commands for a range of kappa values.

    Args:
    input_file (str): Path to PC Scores
    config_data (dict): Dictionary containing model training parameters
    output_dir (str): Path to output directory to save trained models

    Returns:
    command_string (str): CLI command string for model training commands.
    """
    assert (
        "out_script" in config_data
    ), "Need to supply out_script to save modeling commands"

    data_dict, _ = load_pcs(
        filename=input_file,
        var_name=config_data.get("var_name", "scores"),
        npcs=config_data["npcs"],
        load_groups=config_data["load_groups"],
    )

    # Get list of kappa values for spooling models
    kappas = get_scan_range_kappas(data_dict, config_data)

    # Get model training command strings
    command_string = create_command_strings(input_file, output_dir, config_data, kappas)

    # Ensure output directory exists
    os.makedirs(dirname(config_data["out_script"]), exist_ok=True)
    # Write command string to file
    with open(config_data["out_script"], "w") as f:
        f.write(command_string)
    print("Commands saved to:", config_data["out_script"])

    if config_data["get_cmd"]:
        # Display the command string
        print("Listing kappa scan commands...\n")
        print(command_string)
    if config_data["run_cmd"]:
        # Or run the kappa scan
        print("Running kappa scan commands")
        os.system(command_string)

    return command_string


def count_frames_wrapper(input_file):
    # count frames from pc scores
    pass

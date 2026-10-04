"""
GUI front-end functions for training ARHMM.
"""

from moseq2_model.cli import learn_model, kappa_scan_fit_models
from os.path import dirname, join, exists
from moseq2_model.helpers.wrappers import (
    learn_model_wrapper,
    kappa_scan_fit_models_wrapper,
    apply_model_wrapper,
)
from moseq2_model._yaml import read_yaml


def learn_model_command(progress_paths, get_cmd=True, verbose=False):
    """
    Train ARHMM from within a Jupyter notebook using parameters specified in the notebook.

    Args:
    progress_paths (dict): notebook progress dict that contains paths to the pc scores, config, and index files.
    get_cmd (bool): flag to return the kappa scan learn-model commands.
    verbose (bool): compute modeling summary - can slow down training.

    Returns:
    None or kappa scan command
    """

    # Load proper input variables
    input_file = progress_paths["scores_path"]
    dest_file = progress_paths["model_path"]
    config_file = progress_paths["config_file"]
    index = progress_paths["index_file"]
    output_dir = progress_paths["base_model_path"]

    assert exists(
        input_file
    ), "PCA Score not found; set the correct path in progress_paths['scores_path']"
    assert exists(
        config_file
    ), "Config file not found; set the correct path in progress_paths['config_file']"
    assert exists(
        index
    ), "Index file not found; set the correct path in progress_paths['index_file']"

    with open(config_file, "r") as f:
        config_data = read_yaml(f)

    # Get default CLI params
    params = {tmp.name: tmp.default for tmp in learn_model.params if not tmp.required}
    # merge default params and config data, preferring values in config data
    config_data = {**params, **config_data}

    config_data["verbose"] = verbose
    config_data["index"] = index

    if output_dir is None:
        print("Output directory not specified, saving models to base directory.")
        output_dir = dirname(index)

    if config_data["kappa"] == "scan":
        assert config_data["scan_scale"] in (
            "log",
            "linear",
        ), 'scan scale must be "log" or "linear"'

        # Get default CLI params
        params = {
            tmp.name: tmp.default
            for tmp in kappa_scan_fit_models.params
            if not tmp.required
        }
        # merge default params and config data, preferring values in config data
        config_data = {**params, **config_data}

        config_data["out_script"] = join(output_dir, config_data["out_script"])
        config_data["get_cmd"] = get_cmd

        command = kappa_scan_fit_models_wrapper(input_file, config_data, output_dir)
        return command
    else:
        learn_model_wrapper(input_file, dest_file, config_data)


def apply_model_command(progress_paths, model_file):
    """Apply a pre-trained ARHMM to a new dataset from within a Jupyter notebook.

    Args:
        progress_paths (dict): notebook progress dict that contains paths to the pc scores, config, and index files.
        model_file (str): path to the pre-trained ARHMM.
    """

    # Load proper input variables
    pc_file = progress_paths["scores_path"]
    dest_file = progress_paths["model_path"]
    config_file = progress_paths["config_file"]
    index = progress_paths["index_file"]
    output_dir = progress_paths["base_model_path"]

    # load config data
    with open(config_file, "r") as f:
        config_data = read_yaml(f)

    # apply model to data
    apply_model_wrapper(model_file, pc_file, dest_file, config_data)

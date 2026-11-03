"""
GUI front-end functions for training ARHMM.

Each function reads the project's ``config.yaml``, resolves the step's settings
and calls the step in ``moseq2_model.steps``.
"""

from os.path import dirname, join, exists

from moseq2_common.options import read_config
from moseq2_model import steps
from moseq2_model.options import ApplyModelOptions, KappaScanOptions, LearnModelOptions


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

    config = read_config(config_file)

    if output_dir is None:
        print("Output directory not specified, saving models to base directory.")
        output_dir = dirname(index)

    if config.get("kappa") == "scan":
        assert config["scan_scale"] in (
            "log",
            "linear",
        ), 'scan scale must be "log" or "linear"'

        scan = KappaScanOptions.resolve(config=config)
        opts = KappaScanOptions.resolve(
            config=config, explicit={"get_cmd": get_cmd, "out_script": join(output_dir, scan.out_script)}
        )
        return steps.kappa_scan(input_file, output_dir, opts, index=index, shim_flat_config=config).commands
    else:
        opts = LearnModelOptions.resolve(config=config, explicit={"verbose": verbose})
        steps.learn_model(input_file, dest_file, opts, index=index, shim_flat_config=config)


def apply_model_command(progress_paths, model_file):
    """Apply a pre-trained ARHMM to a new dataset from within a Jupyter notebook.

    Args:
        progress_paths (dict): notebook progress dict that contains paths to the pc scores, config, and index files.
        model_file (str): path to the pre-trained ARHMM.
    """

    config = read_config(progress_paths["config_file"])
    # the command line's --load-groups defaults to True; the notebook reads it as False when config.yaml has none
    opts = ApplyModelOptions.resolve(config=config, explicit={"load_groups": config.get("load_groups", False)})

    # apply model to data
    steps.apply_model(
        model_file,
        progress_paths["scores_path"],
        progress_paths["model_path"],
        opts,
        index=None,  # shim L3, removed by PR 4.6
        shim_flat_config=config,
    )

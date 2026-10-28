"""
CLI for modeling the data using AR-HMM.
"""

import os
import click
from os.path import join
from moseq2_model.util import count_frames as count_frames_wrapper
from moseq2_model.helpers.wrappers import (
    learn_model_wrapper,
    kappa_scan_fit_models_wrapper,
    apply_model_wrapper,
)

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass


@cli.command(
    name="count-frames", help="Count the number of frames in given h5 file (pca_scores)"
)
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--var-name",
    type=str,
    default="scores",
    help="Variable name in input file with PCs",
)
def count_frames(input_file, var_name):
    # Count the number of frames in the input file.
    count_frames_wrapper(input_file=input_file, var_name=var_name)


def modeling_parameters(function):

    function = click.option(
        "--hold-out",
        "-h",
        is_flag=True,
        default=False,
        help="Hold out one fold (set by nfolds) for computing heldout likelihood",
    )(function)
    function = click.option(
        "--hold-out-seed",
        type=int,
        default=-1,
        help="Random seed for holding out data (set for reproducibility)",
    )(function)
    function = click.option(
        "--nfolds", type=int, default=5, help="Number of folds for split"
    )(function)
    function = click.option(
        "--ncpus",
        "-c",
        type=int,
        default=0,
        help="Number of cores to use for resampling",
    )(function)
    function = click.option(
        "--num-iter",
        "-n",
        type=int,
        default=100,
        help="Number of interations to resample model",
    )(function)
    function = click.option(
        "--var-name",
        type=str,
        default="scores",
        help="Variable name in input file with PCs",
    )(function)
    function = click.option(
        "--e-step",
        is_flag=True,
        default=False,
        help="Compute the expected state sequence for each recordings",
    )(function)
    function = click.option(
        "--save-every",
        "-s",
        type=int,
        default=-1,
        help="Increment to save labels and model object (-1 for just last)",
    )(function)
    function = click.option(
        "--save-model",
        type=bool,
        default=True,
        help="Save model object at the end of training",
    )(function)
    function = click.option(
        "--max-states", "-m", type=int, default=100, help="Maximum number of states"
    )(function)
    function = click.option(
        "--npcs", type=int, default=10, help="Number of PCs to use"
    )(function)
    function = click.option(
        "--whiten",
        "-w",
        type=str,
        default="all",
        help="Whiten PCs: (e)each session (a)ll combined or (n)o whitening",
    )(function)
    function = click.option(
        "--progressbar", "-p", type=bool, default=True, help="Show model progress"
    )(function)
    function = click.option(
        "--percent-split",
        type=int,
        default=0,
        help="Training-validation split percentage used when not holding out data and when this parameter > 0.",
    )(function)
    function = click.option(
        "--load-groups",
        type=bool,
        default=True,
        help="If groups should be loaded with the PC scores.",
    )(function)
    function = click.option(
        "--gamma",
        "-g",
        type=float,
        default=1e3,
        help="Gamma; hierarchical dirichlet process hyperparameter (try not to change it).",
    )(function)
    function = click.option(
        "--alpha",
        "-a",
        type=float,
        default=5.7,
        help="Alpha; hierarchical dirichlet process hyperparameter (try not to change it).",
    )(function)
    function = click.option(
        "--noise-level",
        type=float,
        default=0,
        help="Additive white gaussian noise to input data for regularization. Not generally used",
    )(function)
    function = click.option(
        "--nlags", type=int, default=3, help="Number of lags to use"
    )(function)
    function = click.option(
        "--separate-trans",
        is_flag=True,
        default=False,
        help="Use separate transition matrix for each group",
    )(function)
    function = click.option(
        "--robust",
        is_flag=True,
        default=False,
        help="Use robust AR-HMM model. More tolerant to noise",
    )(function)
    function = click.option(
        "--check-every",
        type=int,
        default=5,
        help="Increment to record training and validation log-likelihoods.",
    )(function)

    return function


@cli.command(
    name="learn-model", help="Train ARHMM on PC Scores with given training parameters"
)
@click.argument("input_file", type=click.Path(exists=True))
@click.argument(
    "dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True)
)
@modeling_parameters
@click.option(
    "--kappa",
    "-k",
    type=float,
    default=None,
    help="Kappa; hyperparameter used to set syllable duration. Larger k = longer syllable lengths",
)
@click.option(
    "--checkpoint-freq",
    type=int,
    default=-1,
    help="save model checkpoint every n iterations",
)
@click.option(
    "--use-checkpoint",
    is_flag=True,
    default=False,
    help="indicate whether to use previously saved checkpoint",
)
@click.option(
    "--index",
    "-i",
    type=click.Path(),
    default="",
    help="Path to moseq2-index.yaml for group definitions",
)
@click.option(
    "--default-group",
    type=str,
    default="n/a",
    help="Default group name to use for separate-trans",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print syllable log-likelihoods during training.",
)
def learn_model(input_file, dest_file, **config_data):
    # Train the ARHMM using PC scores located in the INPUT_FILE, and saves the model to DEST_FILE

    learn_model_wrapper(input_file, dest_file, config_data)


@cli.command(name="apply-model", help="Apply pre-trained ARHMM to PC scores.")
@click.argument("model_file", type=click.Path(exists=True))
@click.argument("pc_file", type=click.Path(exists=True))
@click.argument(
    "dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True)
)
@click.option(
    "--var-name",
    type=str,
    default="scores",
    help="Variable name in input file with PCs",
)
@click.option(
    "--index",
    "-i",
    type=click.Path(),
    default="",
    help="Path to moseq2-index.yaml for group definitions",
)
@click.option(
    "--load-groups",
    type=bool,
    default=True,
    help="If groups should be loaded with the PC scores.",
)
def apply_model(model_file, pc_file, dest_file, **config_data):
    # Apply the ARHMM located in MODEL_FILE to the PC scores in PC_FILE, and saves the results to DEST_FILE

    apply_model_wrapper(model_file, pc_file, dest_file, config_data)


@cli.command(
    name="kappa-scan",
    help="Batch train multiple model to scan over different kappa values.",
)
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.option(
    "--index",
    "-i",
    type=click.Path(),
    default="",
    help="Path to moseq2-index.yaml for session metadata and group information",
)
@click.option(
    "--out-script",
    type=click.Path(),
    default="train_out.sh",
    help="Name of bash script file to save model training commands.",
)
@click.option(
    "--n-models", type=int, default=10, help="Number of models to train in kappa scan."
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Batch command string to prefix model training command (slurm only).",
)
@click.option(
    "--cluster-type",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Platform to train models on",
)
@click.option(
    "--scan-scale",
    type=click.Choice(["log", "linear"]),
    default="log",
    help="Scale to scan kappa values at.",
)
@click.option(
    "--min-kappa",
    type=float,
    default=None,
    help="Minimum kappa value to begin scan from.",
)
@click.option(
    "--max-kappa", type=float, default=None, help="Maximum kappa value to end scan on."
)
@click.option("--memory", type=str, default="5GB", help="RAM (slurm only)")
@click.option("--wall-time", type=str, default="3:00:00", help="Wall time (slurm only)")
@click.option(
    "--partition", type=str, default="short", help="Partition name (slurm only)"
)
@click.option("--get-cmd", is_flag=True, default=False, help="Print scan command strings.")
@click.option("--run-cmd", is_flag=True, default=False, help="Run scan command strings.")
@modeling_parameters
def kappa_scan_fit_models(input_file, output_dir, **config_data):
    # Scan through the kappa hyperparameter to find the kappa that best matches the changepoint duration distribution.

    config_data["out_script"] = join(output_dir, config_data["out_script"])
    kappa_scan_fit_models_wrapper(input_file, config_data, output_dir)


if __name__ == "__main__":
    cli()

"""The settings of each modeling step, declared once.

One class per command. Each field carries its default, its command-line flags,
its click type and its help text; ``cli.py`` generates the commands' options
from these classes, and ``steps.py`` takes an instance. The files a step works
on (``INPUT_FILE``, ``DEST_FILE``, ``OUTPUT_DIR`` and the like) and ``--index``
are not settings: they are the steps' arguments and plain click parameters in
``cli.py``. ``--index`` is one of them because its help differs between
``learn-model`` and ``kappa-scan``.

The modeling steps' classes read ``config.yaml``'s ``model`` section
(``MODEL_STAGE``).
"""

from __future__ import annotations

from dataclasses import dataclass

import click

from moseq2_common.options import Options, option
from moseq2_common.slurm import SlurmOptions


@dataclass(frozen=True, kw_only=True)
class ModelOptions(Options):
    """The fitting settings ``learn-model`` and ``kappa-scan`` share."""

    SECTION = "model"

    check_every: int = option(5, '--check-every', type=int,
                              help="Increment to record training and validation log-likelihoods.")
    robust: bool = option(False, '--robust', is_flag=True, help="Use robust AR-HMM model. More tolerant to noise")
    separate_trans: bool = option(False, '--separate-trans', is_flag=True,
                                  help="Use separate transition matrix for each group")
    nlags: int = option(3, '--nlags', type=int, help="Number of lags to use")
    noise_level: float = option(0, '--noise-level', type=float,
                                help="Additive white gaussian noise to input data for regularization. "
                                     "Not generally used")
    alpha: float = option(5.7, '--alpha', '-a', type=float,
                          help="Alpha; hierarchical dirichlet process hyperparameter (try not to change it).")
    gamma: float = option(1e3, '--gamma', '-g', type=float,
                          help="Gamma; hierarchical dirichlet process hyperparameter (try not to change it).")
    load_groups: bool = option(True, '--load-groups', type=bool,
                               help="If groups should be loaded with the PC scores.")
    percent_split: int = option(0, '--percent-split', type=int,
                                help="Training-validation split percentage used when not holding out data "
                                     "and when this parameter > 0.")
    progressbar: bool = option(True, '--progressbar', '-p', type=bool, help="Show model progress")
    whiten: str = option('all', '--whiten', '-w', type=str,
                         help="Whiten PCs: (e)each session (a)ll combined or (n)o whitening")
    npcs: int = option(10, '--npcs', type=int, help="Number of PCs to use")
    max_states: int = option(100, '--max-states', '-m', type=int, help="Maximum number of states")
    save_model: bool = option(True, '--save-model', type=bool, help="Save model object at the end of training")
    e_step: bool = option(False, '--e-step', is_flag=True,
                          help="Compute the expected state sequence for each recordings")
    var_name: str = option('scores', '--var-name', type=str, help="Variable name in input file with PCs")
    num_iter: int = option(100, '--num-iter', '-n', type=int, help="Number of interations to resample model")
    ncpus: int = option(0, '--ncpus', '-c', type=int,
                        help="CPUs each kappa-scan job requests on SLURM (sbatch -c; 8 when 0). "
                             "A single fit uses the CPUs the job or machine gives it; this does not change that.")
    nfolds: int = option(5, '--nfolds', type=int, help="Number of folds for split")
    hold_out_seed: int = option(-1, '--hold-out-seed', type=int,
                                help="Random seed for holding out data (set for reproducibility)")
    hold_out: bool = option(False, '--hold-out', '-h', is_flag=True,
                            help="Hold out one fold (set by nfolds) for computing heldout likelihood")


@dataclass(frozen=True, kw_only=True)
class LearnModelOptions(ModelOptions):
    """``learn-model``."""

    kappa: float | None = option(None, '--kappa', '-k', type=float,
                                 help="Kappa; hyperparameter used to set syllable duration. "
                                      "Larger k = longer syllable lengths")
    checkpoint_freq: int = option(-1, '--checkpoint-freq', type=int,
                                  help="save model checkpoint every n iterations")
    use_checkpoint: bool = option(False, '--use-checkpoint', is_flag=True,
                                  help="indicate whether to use previously saved checkpoint")
    default_group: str = option('n/a', '--default-group', type=str,
                                help="Default group name to use for separate-trans")
    verbose: bool = option(False, '--verbose', '-v', is_flag=True, setting=False,
                           help="Print syllable log-likelihoods during training.")


@dataclass(frozen=True, kw_only=True)
class KappaScanOptions(ModelOptions, SlurmOptions):
    """``kappa-scan``: the fitting settings each written ``learn-model`` command gets, the scan and SLURM."""

    out_script: str = option('train_out.sh', '--out-script', type=click.Path(),
                             help="Name of bash script file to save model training commands.")
    n_models: int = option(10, '--n-models', type=int, help="Number of models to train in kappa scan.")
    scan_scale: str = option('log', '--scan-scale', type=click.Choice(['log', 'linear']),
                             help="Scale to scan kappa values at.")
    min_kappa: float | None = option(None, '--min-kappa', type=float,
                                     help="Minimum kappa value to begin scan from.")
    max_kappa: float | None = option(None, '--max-kappa', type=float, help="Maximum kappa value to end scan on.")
    # what the notebook's kappa-scan jobs requested in 1.5.0; batch-extract keeps SlurmOptions' 5GB, 3:00:00
    memory: str = option("15GB", "--memory", type=str, help="RAM (slurm only)")
    wall_time: str = option("06:00:00", "--wall-time", type=str, help="Wall time (slurm only)")


@dataclass(frozen=True, kw_only=True)
class ApplyModelOptions(Options):
    """``apply-model``."""

    SECTION = "model"

    var_name: str = option('scores', '--var-name', type=str, help="Variable name in input file with PCs")
    load_groups: bool = option(True, '--load-groups', type=bool,
                               help="If groups should be loaded with the PC scores.")


@dataclass(frozen=True, kw_only=True)
class CountFramesOptions(Options):
    """``count-frames``."""

    var_name: str = option('scores', '--var-name', type=str, help="Variable name in input file with PCs")


#: The classes of the modeling steps, whose fields config.yaml's ``model`` section may set.
MODEL_STAGE = (LearnModelOptions, KappaScanOptions, ApplyModelOptions)

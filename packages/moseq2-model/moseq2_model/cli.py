"""
CLI for modeling the data using AR-HMM.

Each command's settings are generated from its options class in
``moseq2_model.options``; the command resolves the settings and calls its step
in ``moseq2_model.steps``.
"""

from dataclasses import replace
from os.path import join

import click
from click.core import ParameterSource

from moseq2_common.options import click_options, explicit_values, read_config_option, shim_click_defaults
from moseq2_model import steps
from moseq2_model.options import ApplyModelOptions, CountFramesOptions, KappaScanOptions, LearnModelOptions
from moseq2_model.util import count_frames as count_frames_wrapper


@click.group(context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _index(help):
    """``--index``, which is not a setting; its help differs between commands."""
    return click.option("--index", "-i", type=click.Path(), default="", help=help)


_INDEX_HELP = "Path to moseq2-index.yaml for group definitions"

# The help order of each command, as 1.5.0 lists its options.
_MODEL_ORDER = ("check_every", "robust", "separate_trans", "nlags", "noise_level", "alpha", "gamma", "load_groups",
                "percent_split", "progressbar", "whiten", "npcs", "max_states", "save_model", "e_step",
                "var_name", "num_iter", "ncpus", "nfolds", "hold_out_seed", "hold_out")
LEARN_MODEL_ORDER = ("input_file", "dest_file", *_MODEL_ORDER, "kappa", "checkpoint_freq", "use_checkpoint", "index",
                     "config_file", "default_group", "verbose")
KAPPA_SCAN_ORDER = ("input_file", "output_dir", "index", "out_script", "n_models", "prefix", "cluster_type",
                    "scan_scale", "min_kappa", "max_kappa", "memory", "wall_time", "partition", "get_cmd", "run_cmd",
                    *_MODEL_ORDER)
APPLY_MODEL_ORDER = ("model_file", "pc_file", "dest_file", "var_name", "index", "load_groups")


@cli.command(name="count-frames", help="Count the number of frames in given h5 file (pca_scores)")
@click_options(CountFramesOptions, order=("input_file", "var_name"), others={
    "input_file": click.argument("input_file", type=click.Path(exists=True)),
})
@click.pass_context
def count_frames(ctx, input_file, **_):
    opts = CountFramesOptions.resolve(explicit=explicit_values(ctx, CountFramesOptions))
    count_frames_wrapper(input_file=input_file, var_name=opts.var_name)


@cli.command(name="learn-model", help="Train ARHMM on PC Scores with given training parameters")
@click_options(LearnModelOptions, order=LEARN_MODEL_ORDER, others={
    "input_file": click.argument("input_file", type=click.Path(exists=True)),
    "dest_file": click.argument("dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True)),
    "index": _index(_INDEX_HELP),
    "config_file": click.option("--config-file", type=click.Path(), default=None,
                                help="Path to a config.yaml whose settings this fit uses; "
                                     "options typed on the command line take precedence over it"),
})
@click.pass_context
def learn_model(ctx, input_file, dest_file, index, config_file, **_):
    # Train the ARHMM using PC scores located in the INPUT_FILE, and saves the model to DEST_FILE
    opts = shim_click_defaults(ctx, LearnModelOptions, read_config_option(config_file))
    if opts.kappa is not None and (isinstance(opts.kappa, bool) or not isinstance(opts.kappa, (int, float))):
        where = "on the command line" if ctx.get_parameter_source("kappa") is ParameterSource.COMMANDLINE \
            else f"in {config_file}"
        raise click.ClickException(
            f"kappa is {opts.kappa!r} {where}, and learn-model fits one model at one kappa. Set kappa to a "
            "number (or delete it to use the number of frames), or run moseq2-model kappa-scan to fit a "
            "range of kappas."
        )
    if ctx.get_parameter_source("ncpus") is ParameterSource.COMMANDLINE and opts.ncpus:
        click.echo(
            "--ncpus sets the CPUs each kappa-scan job requests on SLURM. A single fit "
            "uses the CPUs the job or machine gives it, and --ncpus does not change that."
        )
    steps.learn_model(input_file, dest_file, opts, index=index)


@cli.command(name="apply-model", help="Apply pre-trained ARHMM to PC scores.")
@click_options(ApplyModelOptions, order=APPLY_MODEL_ORDER, others={
    "model_file": click.argument("model_file", type=click.Path(exists=True)),
    "pc_file": click.argument("pc_file", type=click.Path(exists=True)),
    "dest_file": click.argument("dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True)),
    "index": _index(_INDEX_HELP),
})
@click.pass_context
def apply_model(ctx, model_file, pc_file, dest_file, index, **_):
    # Apply the ARHMM located in MODEL_FILE to the PC scores in PC_FILE, and saves the results to DEST_FILE
    opts = ApplyModelOptions.resolve(explicit=explicit_values(ctx, ApplyModelOptions))
    steps.apply_model(model_file, pc_file, dest_file, opts, index=index)


@cli.command(name="kappa-scan", help="Batch train multiple models to scan over different kappa values. Each "
             "model's learn-model settings are written to a file beside it, which its command passes to "
             "learn-model --config-file.")
@click_options(KappaScanOptions, order=KAPPA_SCAN_ORDER, others={
    "input_file": click.argument("input_file", type=click.Path(exists=True)),
    "output_dir": click.argument("output_dir", type=click.Path(exists=False)),
    "index": _index("Path to moseq2-index.yaml for session metadata and group information"),
})
@click.pass_context
def kappa_scan_fit_models(ctx, input_file, output_dir, index, **_):
    # Scan through the kappa hyperparameter to find the kappa that best matches the changepoint duration distribution.
    opts = shim_click_defaults(ctx, KappaScanOptions)
    steps.kappa_scan(input_file, output_dir, replace(opts, out_script=join(output_dir, opts.out_script)), index=index)


if __name__ == "__main__":
    cli()

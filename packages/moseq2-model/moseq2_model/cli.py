"""
CLI for modeling the data using AR-HMM.

Each command's settings are generated from its options class in
``moseq2_model.options``. A command runs on one project: ``--project-file``'s,
else the project ``--config-file`` belongs to (a kappa-scan fit's settings file
names its project), else the working folder's. It reads the ``model`` section of
``--config-file`` or, without one, of the project's ``config.yaml``, resolves the
settings and calls its step in ``moseq2_model.steps``, with the sessions and
their groups from the project's ``moseq2-project.yaml`` and ``config.yaml``.
"""

from dataclasses import replace
from os.path import join

import click
from click.core import ParameterSource

from moseq2_common.bookkeeping import project_index
from moseq2_common.options import (
    ProjectGroup,
    click_options,
    command_project,
    command_settings,
    described_argument,
    explicit_values,
    project_file_option,
)
from moseq2_common.project import is_job_file
from moseq2_common.records import print_record_command
from moseq2_model import steps
from moseq2_model.options import (
    MODEL_STAGE,
    ApplyModelOptions,
    CountFramesOptions,
    KappaScanOptions,
    LearnModelOptions,
)
from moseq2_model.util import count_frames as count_frames_wrapper


@click.group(cls=ProjectGroup, context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


#: Where the PC scores and a fitted model usually are, for the messages of a missing argument.
_SCORES = "the .h5 apply-pca writes (_pca/pca_scores.h5 in a project)"
_LEARN_EXAMPLE = "For example, from the project folder: `moseq2-model learn-model _pca/pca_scores.h5 models/model.p`."


def _scores(what):
    return described_argument("input_file", type=click.Path(exists=True), missing=f"the PC scores {what}, {_SCORES}.")


_PROJECT_FILE_HELP = ("The project's moseq2-project.yaml, whose sessions and groups (in its config.yaml) the "
                      "command uses; by default the project in the folder the command runs in")

# The help order of each command, as 1.5.0 lists its options.
_MODEL_ORDER = ("check_every", "robust", "separate_trans", "nlags", "noise_level", "alpha", "gamma", "load_groups",
                "percent_split", "progressbar", "whiten", "npcs", "max_states", "save_model", "e_step",
                "var_name", "num_iter", "ncpus", "nfolds", "hold_out_seed", "hold_out")
LEARN_MODEL_ORDER = ("input_file", "dest_file", *_MODEL_ORDER, "kappa", "checkpoint_freq", "use_checkpoint",
                     "project_file",
                     "config_file", "default_group", "verbose")
KAPPA_SCAN_ORDER = ("input_file", "output_dir", "project_file", "out_script", "n_models", "prefix", "cluster_type",
                    "scan_scale", "min_kappa", "max_kappa", "memory", "wall_time", "partition", "get_cmd", "run_cmd",
                    *_MODEL_ORDER)
APPLY_MODEL_ORDER = ("model_file", "pc_file", "dest_file", "var_name", "project_file", "load_groups")


@cli.command(name="count-frames", help="Count the number of frames in given h5 file (pca_scores)")
@click_options(CountFramesOptions, order=("input_file", "var_name"), others={
    "input_file": described_argument("input_file", type=click.Path(exists=True),
                                     missing=f"the PC scores file whose frames to count, {_SCORES}."),
})
@click.pass_context
def count_frames(ctx, input_file, **_):
    opts = CountFramesOptions.resolve(explicit=explicit_values(ctx, CountFramesOptions))
    count_frames_wrapper(input_file=input_file, var_name=opts.var_name)


@cli.command(name="learn-model", help="Train ARHMM on PC Scores with given training parameters")
@click_options(LearnModelOptions, order=LEARN_MODEL_ORDER, others={
    "input_file": described_argument("input_file", type=click.Path(exists=True),
                                     missing=f"the PC scores to fit the model to, {_SCORES}. {_LEARN_EXAMPLE}"),
    "dest_file": described_argument("dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True),
                                    missing="the file to write the fitted model to, a .p file (the notebook writes "
                                            f"models/model.p in the project folder). {_LEARN_EXAMPLE}"),
    "project_file": project_file_option(_PROJECT_FILE_HELP),
    "config_file": click.option("--config-file", type=click.Path(), default=None,
                                help="Path to a config.yaml whose settings this fit uses; "
                                     "options typed on the command line take precedence over it"),
})
@click.pass_context
def learn_model(ctx, input_file, dest_file, config_file, **_):
    # Train the ARHMM using PC scores located in the INPUT_FILE, and saves the model to DEST_FILE
    config = command_settings(ctx, LearnModelOptions, stage_classes=MODEL_STAGE)
    opts = LearnModelOptions.resolve(config=config, explicit=explicit_values(ctx, LearnModelOptions))
    if opts.kappa is not None and (isinstance(opts.kappa, bool) or not isinstance(opts.kappa, (int, float))):
        where = "on the command line" if ctx.get_parameter_source("kappa") is ParameterSource.COMMANDLINE \
            else f"in the model section of {config_file or 'config.yaml'}"
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
    project = command_project(ctx)
    if is_job_file(config_file):
        # a kappa-scan fit: its settings file names the project, whose bookkeeping only the scan's submitter writes
        steps.learn_model(input_file, dest_file, opts, index=project_index(project))
    else:
        steps.learn_model(input_file, dest_file, opts, index=project_index(project), project=project)


@cli.command(name="apply-model", help="Apply pre-trained ARHMM to PC scores.")
@click_options(ApplyModelOptions, order=APPLY_MODEL_ORDER, others={
    "model_file": described_argument("model_file", type=click.Path(exists=True),
                                     missing="the fitted model to apply, the .p file learn-model writes "
                                             "(models/model.p in a project)."),
    "pc_file": described_argument("pc_file", type=click.Path(exists=True),
                                  missing=f"the PC scores to label with the model, {_SCORES}."),
    "dest_file": described_argument("dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True),
                                    missing="the file to write the labels to, a .p file (for example "
                                            "models/applied_model.p in the project folder)."),
    "project_file": project_file_option(_PROJECT_FILE_HELP),
})
@click.pass_context
def apply_model(ctx, model_file, pc_file, dest_file, **_):
    # Apply the ARHMM located in MODEL_FILE to the PC scores in PC_FILE, and saves the results to DEST_FILE
    config = command_settings(ctx, ApplyModelOptions, stage_classes=MODEL_STAGE)
    opts = ApplyModelOptions.resolve(config=config, explicit=explicit_values(ctx, ApplyModelOptions))
    project = command_project(ctx)
    steps.apply_model(model_file, pc_file, dest_file, opts, index=project_index(project), project=project)


@cli.command(name="kappa-scan", help="Batch train multiple models to scan over different kappa values. Each "
             "model's learn-model settings are written to a file beside it (model-000-1000-job.yaml for "
             "model-000-1000.p), which its command passes to learn-model --config-file. The models enter "
             "the project's moseq2-project.yaml when get-best-model compares them.")
@click_options(KappaScanOptions, order=KAPPA_SCAN_ORDER, others={
    "input_file": _scores("to fit the models to"),
    "output_dir": described_argument("output_dir", type=click.Path(exists=False),
                                     missing="the folder to write the scan's models and their job files into "
                                             "(for example models/kappa_scan in the project folder), which "
                                             "moseq2-viz get-best-model then compares."),
    "project_file": project_file_option(_PROJECT_FILE_HELP),
})
@click.pass_context
def kappa_scan_fit_models(ctx, input_file, output_dir, **_):
    # Scan through the kappa hyperparameter to find the kappa that best matches the changepoint duration distribution.
    config = command_settings(ctx, KappaScanOptions, stage_classes=MODEL_STAGE)
    opts = KappaScanOptions.resolve(config=config, explicit=explicit_values(ctx, KappaScanOptions))
    learn = LearnModelOptions.resolve(config=config)
    project = command_project(ctx)
    steps.kappa_scan(input_file, output_dir, replace(opts, out_script=join(output_dir, opts.out_script)),
                     index=project_index(project), learn=learn, project=project)


cli.add_command(print_record_command())


if __name__ == "__main__":
    cli()

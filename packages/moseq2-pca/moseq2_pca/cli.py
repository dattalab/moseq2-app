"""
CLI for PCA and model-free changepoint analysis.

Each command's settings are generated from its options class in
``moseq2_pca.options``; the command reads ``--config-file``, resolves the
settings and calls its step in ``moseq2_pca.steps``. Inside a project (the
folder of ``--config-file``, or the working folder when it holds a
``config.yaml``), each command writes into the project's ``_pca/`` unless
``--output-dir`` is given, records what it wrote in the project's
``moseq2-project.yaml``, and ``apply-pca`` and ``compute-changepoints`` read the
components and scores from there unless a path is given. A typed
``--input-dir`` is the folder of extractions the command reads, inside a project
or not; without one, a command reads the sessions the project's
``moseq2-project.yaml`` lists (less ``exclude``), or, outside a project, the
extractions in the working folder. Outside a project, ``--output-dir`` defaults
to the working folder's ``_pca/``. ``--exclude`` and ``--no-exclude`` replace
``config.yaml``'s ``exclude`` for one run, also for a typed ``--input-dir``.
``clip-scores`` reads the project's PC scores when its ``PCA_FILE`` is left out.

A command with nothing to read stops before any work, naming where it looked
and the ways out (``moseq2_pca.steps``). ``train-pca`` names each trained
setting typed at another value than ``config.yaml``'s, on which ``apply-pca``
and ``compute-changepoints`` will stop.
"""

import os
from os.path import join

import click
from moseq2_common.bookkeeping import BookkeepingError, load, resolve_path, sessions_folder
from moseq2_common.options import (
    ProjectGroup,
    click_options,
    command_project,
    PROJECT_SCORES,
    command_settings,
    exclude_options,
    explicit_values,
    output_folder,
    project_argument,
    typed_exclude,
)
from moseq2_common.records import print_record_command
from moseq2_pca import steps
from moseq2_pca.helpers.data import TrainedSettingsError
from moseq2_pca.helpers.wrappers import clip_scores_wrapper
from moseq2_pca.options import PCA_STAGE, ApplyPCAOptions, ChangepointsOptions, ClipScoresOptions, TrainPCAOptions


@click.group(cls=ProjectGroup, context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _fill_gaps_is_not_an_option(ctx, param, value):
    """Stop on --fill-gaps, saying why dropped frames are always filled."""
    if value is not None:
        raise click.UsageError(
            f'apply-pca does not take --fill-gaps (given: --fill-gaps {value}). Dropped frames are '
            'always filled with empty (NaN) rows when a session has timestamps: changepoints and '
            'the model treat each row of PC scores as one frame step, so leaving the rows out '
            'would make them assume no frame was dropped. Remove --fill-gaps from the command.',
            ctx=ctx,
        )


def _paths(output_file):
    """The options that are not settings: the step's paths and the config file."""
    return {
        'input_dir': click.option('--input-dir', '-i', type=click.Path(), default=None, show_default=False,
                                  help='Directory to find extracted h5 files, read instead of the project\'s '
                                       'sessions (default: the sessions moseq2-project.yaml lists, or the folder '
                                       'the command runs in outside a project)'),
        'output_dir': click.option('--output-dir', '-o', default=None, show_default=False, type=click.Path(),
                                   help='Directory to store PCA results (default: _pca in the project folder, '
                                        'or in the folder the command runs in outside a project)'),
        'output_file': click.option('--output-file', default=output_file, type=str,
                                    help='Name of h5 file for storing pca results'),
        'config_file': click.option('--config-file', type=click.Path(), default=None,
                                    help="Path to configuration file"),
        **exclude_options(),
    }


#: Where the PC scores usually are, for the message of a missing argument.
SCORES_FILE = "the .h5 apply-pca writes (_pca/pca_scores.h5 in a project)"

# The help order of each command, as 1.5.0 lists its options.
_INPUT_AND_DASK = ('chunk_size', 'h5_mask_path', 'h5_path', 'config_file', 'output_dir', 'input_dir',
                   'cluster_type', 'timeout', 'wall_time', 'memory', 'processes', 'cores', 'nworkers',
                   'queue', 'dask_port', 'dask_cache_path')
TRAIN_PCA_ORDER = (*_INPUT_AND_DASK, 'gaussfilter_space', 'gaussfilter_time', 'medfilter_space',
                   'medfilter_time', 'missing_data', 'missing_data_iters', 'mask_threshold',
                   'mask_height_threshold', 'min_height', 'max_height', 'tailfilter_size', 'tailfilter_shape',
                   'use_fft', 'train_on_subset', 'recon_pcs', 'rank', 'output_file', 'local_processes',
                   'overwrite_pca_train', 'exclude', 'no_exclude')
APPLY_PCA_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_path', 'pca_file', 'fill_gaps', 'verbose',
                   'overwrite_pca_apply', 'exclude', 'no_exclude')
CHANGEPOINTS_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_file', 'pca_file_scores', 'pca_path',
                      'neighbors', 'threshold', 'klags', 'sigma', 'dims', 'verbose', 'exclude', 'no_exclude')


def _folders(ctx, input_dir, output_dir):
    """The project the command runs in (None outside one), the folder of extractions it reads and its output
    folder. The folder read is ``--input-dir``; without it, None in a project (its sessions, which must have been
    aggregated), else the working folder. The output folder is ``--output-dir``, else ``_pca/`` in the project
    folder (in the working folder outside a project)."""
    project = command_project(ctx)
    if not input_dir and project is not None:
        try:
            sessions_folder(project)
        except BookkeepingError as error:
            raise BookkeepingError(f'{error} To read a folder of extractions instead, pass it with --input-dir.') \
                from None
    elif not input_dir:
        input_dir = os.getcwd()
    return project, input_dir or None, output_dir or join(output_folder(ctx), '_pca')


def recorded_paths(project) -> dict:
    """The components, scores and changepoints files the project's bookkeeping file names (``pca_file``,
    ``pca_file_scores``, ``changepoint_file``); empty outside a project."""
    if project is None:
        return {}
    stored = load(project).pca
    names = {'components': 'pca_file', 'scores': 'pca_file_scores', 'changepoints': 'changepoint_file'}
    return {names[output]: str(resolve_path(project, path)) for output, path in stored.items() if output in names}


@cli.command(name='train-pca', help='Train PCA on all extracted results (h5 files) in input directory')
@click_options(TrainPCAOptions, order=TRAIN_PCA_ORDER, others=_paths('pca'))
@click.pass_context
def train_pca(ctx, input_dir, output_dir, output_file, **_):
    config = command_settings(ctx, TrainPCAOptions, stage_classes=PCA_STAGE)
    typed = explicit_values(ctx, TrainPCAOptions)
    opts = TrainPCAOptions.resolve(config=config, explicit=typed)
    project, input_dir, output_dir = _folders(ctx, input_dir, output_dir)
    config_file = ctx.params.get('config_file') or (project.config_file if project is not None else None)
    for line in steps.trained_settings_heads_up({key: getattr(opts, key) for key in typed}, config, config_file):
        click.echo(line)
    steps.train_pca(input_dir, output_dir, output_file, opts, project=project, exclude=typed_exclude(ctx))


@cli.command(name='apply-pca', help='Compute PCA Scores of extraction data given a pre-trained PCA')
@click_options(ApplyPCAOptions, order=APPLY_PCA_ORDER, others={
    **_paths('pca_scores'),
    'pca_file': click.option('--pca-file', type=click.Path(), default=None, help='Path to PCA results'),
    'fill_gaps': click.option('--fill-gaps', hidden=True, expose_value=False, default=None, type=str,
                              callback=_fill_gaps_is_not_an_option),
})
@click.pass_context
def apply_pca(ctx, input_dir, output_dir, output_file, pca_file, **_):
    config = command_settings(ctx, ApplyPCAOptions, stage_classes=PCA_STAGE)
    opts = ApplyPCAOptions.resolve(config=config, explicit=explicit_values(ctx, ApplyPCAOptions))
    project, input_dir, output_dir = _folders(ctx, input_dir, output_dir)
    try:
        steps.apply_pca(input_dir, output_dir, output_file, opts,
                        components_file=pca_file or recorded_paths(project).get('pca_file'), project=project,
                        exclude=typed_exclude(ctx))
    except TrainedSettingsError as error:
        raise click.ClickException(str(error)) from None


@cli.command('compute-changepoints',
             help='Compute the Model-Free Syllable Changepoints based on the PCA/PCA_Scores')
@click_options(ChangepointsOptions, order=CHANGEPOINTS_ORDER, others={
    **_paths('changepoints'),
    'pca_file': click.option('--pca-file', type=click.Path(), default=None, show_default=False,
                             help='The PCA components file train-pca wrote (default: the components '
                                  'moseq2-project.yaml names, or pca.h5 in the output folder outside a project)'),
    'pca_file_scores': click.option('--pca-file-scores', type=click.Path(), default=None,
                                    help='Path to PCA results'),
})
@click.pass_context
def compute_changepoints(ctx, input_dir, output_dir, output_file, pca_file, pca_file_scores, **_):
    config = command_settings(ctx, ChangepointsOptions, stage_classes=PCA_STAGE)
    opts = ChangepointsOptions.resolve(config=config, explicit=explicit_values(ctx, ChangepointsOptions))
    project, input_dir, output_dir = _folders(ctx, input_dir, output_dir)
    recorded = recorded_paths(project)
    try:
        steps.compute_changepoints(input_dir, output_dir, output_file, opts,
                                   components_file=pca_file or recorded.get('pca_file'),
                                   scores_file=pca_file_scores or recorded.get('pca_file_scores'), project=project,
                                   exclude=typed_exclude(ctx))
    except TrainedSettingsError as error:
        raise click.ClickException(str(error)) from None


@cli.command('clip-scores', help='Clip specified number of frames from PCA scores at the beginning or end')
@click_options(ClipScoresOptions, order=('pca_file', 'clip_samples', 'from_end'), others={
    'pca_file': project_argument('pca_file', type=click.Path(exists=True, resolve_path=True),
                                 missing=f'the PC scores file to clip, {SCORES_FILE}.', default=PROJECT_SCORES),
    'clip_samples': project_argument('clip_samples', type=int,
                                     missing="the number of frames to clip from the start of each session's "
                                             "scores (from the end with --from-end)."),
})
@click.pass_context
def clip_scores(ctx, pca_file, clip_samples, **_):
    opts = ClipScoresOptions.resolve(explicit=explicit_values(ctx, ClipScoresOptions))
    clip_scores_wrapper(pca_file, clip_samples, opts.from_end)


cli.add_command(print_record_command())


if __name__ == '__main__':
    cli()

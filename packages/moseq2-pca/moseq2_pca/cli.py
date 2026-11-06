"""
CLI for PCA and model-free changepoint analysis.

Each command's settings are generated from its options class in
``moseq2_pca.options``; the command reads ``--config-file``, resolves the
settings and calls its step in ``moseq2_pca.steps``. Inside a project (the
folder of ``--config-file``, or the working folder when it holds a
``config.yaml``), each command records what it wrote in the project's
``moseq2-project.yaml``, and ``apply-pca`` and ``compute-changepoints`` read the
components and scores from there unless a path is given. Inside a project, the
commands read the sessions ``moseq2-project.yaml`` lists (less ``exclude``), not
``--input-dir``, which only a folder of extractions outside a project uses.
"""

import os
from os.path import join

import click
from click.core import ParameterSource

from moseq2_common.bookkeeping import load, resolve_path
from moseq2_common.options import ProjectGroup, click_options, command_settings, explicit_values
from moseq2_common.project import find_project
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
        'input_dir': click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(),
                                  help='Directory to find extracted h5 files'),
        'output_dir': click.option('--output-dir', '-o', default=join(os.getcwd(), '_pca'), type=click.Path(),
                                   help='Directory to store PCA results'),
        'output_file': click.option('--output-file', default=output_file, type=str,
                                    help='Name of h5 file for storing pca results'),
        'config_file': click.option('--config-file', type=click.Path(), default=None,
                                    help="Path to configuration file"),
    }


# The help order of each command, as 1.5.0 lists its options.
_INPUT_AND_DASK = ('chunk_size', 'h5_mask_path', 'h5_path', 'config_file', 'output_dir', 'input_dir',
                   'cluster_type', 'timeout', 'wall_time', 'memory', 'processes', 'cores', 'nworkers',
                   'queue', 'dask_port', 'dask_cache_path')
TRAIN_PCA_ORDER = (*_INPUT_AND_DASK, 'gaussfilter_space', 'gaussfilter_time', 'medfilter_space',
                   'medfilter_time', 'missing_data', 'missing_data_iters', 'mask_threshold',
                   'mask_height_threshold', 'min_height', 'max_height', 'tailfilter_size', 'tailfilter_shape',
                   'use_fft', 'train_on_subset', 'recon_pcs', 'rank', 'output_file', 'local_processes',
                   'overwrite_pca_train')
APPLY_PCA_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_path', 'pca_file', 'fill_gaps', 'verbose',
                   'overwrite_pca_apply')
CHANGEPOINTS_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_file_scores', 'pca_path',
                      'neighbors', 'threshold', 'klags', 'sigma', 'dims', 'verbose')


def _project_or_input_dir(ctx, config_file, input_dir):
    """The project the command runs in (None outside one), saying so when a typed --input-dir goes unread."""
    project = find_project(config_file)
    if project is not None and ctx.get_parameter_source('input_dir') is ParameterSource.COMMANDLINE:
        click.echo(f'{ctx.info_name} runs in the project {project.folder}, so it reads the sessions '
                   f'{project.bookkeeping_file} lists, not --input-dir {input_dir}.')
    return project


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
def train_pca(ctx, input_dir, output_dir, output_file, config_file, **_):
    config = command_settings(config_file, TrainPCAOptions, stage_classes=PCA_STAGE)
    opts = TrainPCAOptions.resolve(config=config, explicit=explicit_values(ctx, TrainPCAOptions))
    steps.train_pca(input_dir, output_dir, output_file, opts,
                    project=_project_or_input_dir(ctx, config_file, input_dir))


@cli.command(name='apply-pca', help='Compute PCA Scores of extraction data given a pre-trained PCA')
@click_options(ApplyPCAOptions, order=APPLY_PCA_ORDER, others={
    **_paths('pca_scores'),
    'pca_file': click.option('--pca-file', type=click.Path(), default=None, help='Path to PCA results'),
    'fill_gaps': click.option('--fill-gaps', hidden=True, expose_value=False, default=None, type=str,
                              callback=_fill_gaps_is_not_an_option),
})
@click.pass_context
def apply_pca(ctx, input_dir, output_dir, output_file, config_file, pca_file, **_):
    config = command_settings(config_file, ApplyPCAOptions, stage_classes=PCA_STAGE)
    opts = ApplyPCAOptions.resolve(config=config, explicit=explicit_values(ctx, ApplyPCAOptions))
    project = _project_or_input_dir(ctx, config_file, input_dir)
    try:
        steps.apply_pca(input_dir, output_dir, output_file, opts,
                        components_file=pca_file or recorded_paths(project).get('pca_file'), project=project)
    except TrainedSettingsError as error:
        raise click.ClickException(str(error)) from None


@cli.command('compute-changepoints',
             help='Compute the Model-Free Syllable Changepoints based on the PCA/PCA_Scores')
@click_options(ChangepointsOptions, order=CHANGEPOINTS_ORDER, others={
    **_paths('changepoints'),
    'pca_file_scores': click.option('--pca-file-scores', type=click.Path(), default=None,
                                    help='Path to PCA results'),
})
@click.pass_context
def compute_changepoints(ctx, input_dir, output_dir, output_file, config_file, pca_file_scores, **_):
    config = command_settings(config_file, ChangepointsOptions, stage_classes=PCA_STAGE)
    opts = ChangepointsOptions.resolve(config=config, explicit=explicit_values(ctx, ChangepointsOptions))
    project = _project_or_input_dir(ctx, config_file, input_dir)
    recorded = recorded_paths(project)
    try:
        steps.compute_changepoints(input_dir, output_dir, output_file, opts,
                                   components_file=recorded.get('pca_file'),
                                   scores_file=pca_file_scores or recorded.get('pca_file_scores'), project=project)
    except TrainedSettingsError as error:
        raise click.ClickException(str(error)) from None


@cli.command('clip-scores', help='Clip specified number of frames from PCA scores at the beginning or end')
@click_options(ClipScoresOptions, order=('pca_file', 'clip_samples', 'from_end'), others={
    'pca_file': click.argument('pca_file', type=click.Path(exists=True, resolve_path=True)),
    'clip_samples': click.argument('clip_samples', type=int),
})
@click.pass_context
def clip_scores(ctx, pca_file, clip_samples, **_):
    opts = ClipScoresOptions.resolve(explicit=explicit_values(ctx, ClipScoresOptions))
    clip_scores_wrapper(pca_file, clip_samples, opts.from_end)


cli.add_command(print_record_command())


if __name__ == '__main__':
    cli()

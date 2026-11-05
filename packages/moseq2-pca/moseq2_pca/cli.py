"""
CLI for PCA and model-free changepoint analysis.

Each command's settings are generated from its options class in
``moseq2_pca.options``; the command reads ``--config-file``, resolves the
settings and calls its step in ``moseq2_pca.steps``.
"""

import os
from os.path import join

import click

from moseq2_common.options import click_options, command_settings, explicit_values, read_config
from moseq2_common.yaml import write_config
from moseq2_pca import steps
from moseq2_pca.helpers.wrappers import clip_scores_wrapper
from moseq2_pca.options import PCA_STAGE, ApplyPCAOptions, ChangepointsOptions, ClipScoresOptions, TrainPCAOptions


@click.group(context_settings={"show_default": True})
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
APPLY_PCA_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_path', 'pca_file', 'fill_gaps', 'fps', 'verbose',
                   'overwrite_pca_apply')
CHANGEPOINTS_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_file_scores', 'pca_path',
                      'neighbors', 'threshold', 'klags', 'sigma', 'dims', 'fps', 'verbose')


#: The output path each PCA step records at config.yaml's top level, by step.
_WRITTEN_BACK = {'train_pca': 'pca_file', 'apply_pca': 'pca_file_scores', 'compute_changepoints': 'changepoint_file'}


def shim_write_back_paths(config_file, step, written):
    """Record the file a PCA step wrote at ``config_file``'s top level (L12, removed by PR 4.3).

    The next PCA step's front end reads it there (``recorded_paths``), as 1.5.0
    found the components and scores through config.yaml. No resolver reads the
    top level, and PR 4.3's bookkeeping file replaces it. Nothing is written
    without a config file, or when the step wrote nothing.
    """
    if not config_file or written is None:
        return
    data = read_config(config_file)
    data[_WRITTEN_BACK[step]] = str(written)
    write_config(data, config_file)


def recorded_paths(config_file) -> dict:
    """The components, scores and changepoints files earlier PCA steps recorded in ``config_file``."""
    if not config_file:
        return {}
    data = read_config(config_file)
    return {key: data[key] for key in _WRITTEN_BACK.values() if data.get(key)}


def _config_file(config_file):
    """``--config-file``, or the config.yaml of the project the command runs in."""
    from moseq2_common.project import find_project

    if config_file:
        return config_file
    project = find_project()
    return str(project.config_file) if project is not None else None


@cli.command(name='train-pca', help='Train PCA on all extracted results (h5 files) in input directory')
@click_options(TrainPCAOptions, order=TRAIN_PCA_ORDER, others=_paths('pca'))
@click.pass_context
def train_pca(ctx, input_dir, output_dir, output_file, config_file, **_):
    config = command_settings(config_file, TrainPCAOptions, stage_classes=PCA_STAGE)
    opts = TrainPCAOptions.resolve(config=config, explicit=explicit_values(ctx, TrainPCAOptions))
    result = steps.train_pca(input_dir, output_dir, output_file, opts)
    shim_write_back_paths(_config_file(config_file), 'train_pca', result.components_file)


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
    recorded = recorded_paths(_config_file(config_file))
    result = steps.apply_pca(input_dir, output_dir, output_file, opts,
                             components_file=pca_file or recorded.get('pca_file'))
    shim_write_back_paths(_config_file(config_file), 'apply_pca', result.scores_file)


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
    recorded = recorded_paths(_config_file(config_file))
    result = steps.compute_changepoints(input_dir, output_dir, output_file, opts,
                                        components_file=recorded.get('pca_file'),
                                        scores_file=pca_file_scores or recorded.get('pca_file_scores'))
    shim_write_back_paths(_config_file(config_file), 'compute_changepoints', result.changepoints_file)


@cli.command('clip-scores', help='Clip specified number of frames from PCA scores at the beginning or end')
@click_options(ClipScoresOptions, order=('pca_file', 'clip_samples', 'from_end'), others={
    'pca_file': click.argument('pca_file', type=click.Path(exists=True, resolve_path=True)),
    'clip_samples': click.argument('clip_samples', type=int),
})
@click.pass_context
def clip_scores(ctx, pca_file, clip_samples, **_):
    opts = ClipScoresOptions.resolve(explicit=explicit_values(ctx, ClipScoresOptions))
    clip_scores_wrapper(pca_file, clip_samples, opts.from_end)


if __name__ == '__main__':
    cli()

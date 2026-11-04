"""
CLI for PCA and model-free changepoint analysis.

Each command's settings are generated from its options class in
``moseq2_pca.options``; the command reads ``--config-file``, resolves the
settings and calls its step in ``moseq2_pca.steps``.
"""

import os
from dataclasses import replace
from os.path import join

import click

from moseq2_common.options import click_options, explicit_values, read_config_option, shim_click_defaults
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_pca import steps
from moseq2_pca.helpers.wrappers import clip_scores_wrapper
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, ClipScoresOptions, TrainPCAOptions


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
                   'overwrite_pca_train', 'camera_type')
APPLY_PCA_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_path', 'pca_file', 'fill_gaps', 'fps', 'detrend_window',
                   'verbose', 'overwrite_pca_apply')
CHANGEPOINTS_ORDER = (*_INPUT_AND_DASK, 'output_file', 'pca_file_components', 'pca_file_scores', 'pca_path',
                      'neighbors', 'threshold', 'klags', 'sigma', 'dims', 'fps', 'verbose')


# 1.5.0's command_with_config left these out of what it wrote back.
_NOT_WRITTEN_BACK = ('missing_data', 'use_fft', 'verbose', 'from_end')


def shim_write_back_before(ctx, config_file, config, opts):
    """Write the settings a command runs with into ``config_file`` before it runs (L12, removed by PR 4.3).

    As 1.5.0's ``command_with_config``: the file, then every parameter of the
    command (paths included, as resolved) except ``_NOT_WRITTEN_BACK``.
    """
    if not config_file:
        return
    resolved = opts.as_dict()
    data = {**config, 'input_dir': None, 'output_dir': None, 'output_file': None}
    data.update({name: resolved.get(name, value) for name, value in ctx.params.items()
                 if name not in _NOT_WRITTEN_BACK})
    write_yaml(data, config_file)


def shim_write_back_after(config_file, result):
    """Merge what the step returned into ``config_file`` (L12, removed by PR 4.3).

    As 1.5.0's ``combine_new_config``: everything but ``output_dir`` and ``output_file``.
    """
    if not config_file:
        return
    data = {**read_yaml(config_file), **result.shim_config_data}
    write_yaml({k: v for k, v in data.items() if k not in ('output_dir', 'output_file')}, config_file)


def shim_camera_swap(opts: TrainPCAOptions) -> TrainPCAOptions:
    """Azure filters in place of the Kinect defaults when ``camera_type`` is azure (L6, removed by PR 4.1)."""
    if opts.camera_type != 'azure':
        return opts
    print('Updating parameters for Azure Kinect camera...')
    if opts.gaussfilter_space == (1.5, 1):
        opts = replace(opts, gaussfilter_space=(2.25, 1.5))
    if opts.tailfilter_size == (9, 9):
        opts = replace(opts, tailfilter_size=(13, 13))
    return opts


@cli.command(name='train-pca', help='Train PCA on all extracted results (h5 files) in input directory')
@click_options(TrainPCAOptions, order=TRAIN_PCA_ORDER, others=_paths('pca'))
@click.pass_context
def train_pca(ctx, input_dir, output_dir, output_file, config_file, **_):
    config = read_config_option(config_file)
    opts = shim_click_defaults(ctx, TrainPCAOptions, config)
    shim_write_back_before(ctx, config_file, config, opts)
    result = steps.train_pca(input_dir, output_dir, output_file, shim_camera_swap(opts))
    shim_write_back_after(config_file, result)


@cli.command(name='apply-pca', help='Compute PCA Scores of extraction data given a pre-trained PCA')
@click_options(ApplyPCAOptions, order=APPLY_PCA_ORDER, others={
    **_paths('pca_scores'),
    'fill_gaps': click.option('--fill-gaps', hidden=True, expose_value=False, default=None, type=str,
                              callback=_fill_gaps_is_not_an_option),
})
@click.pass_context
def apply_pca(ctx, input_dir, output_dir, output_file, config_file, **_):
    config = read_config_option(config_file)
    opts = shim_click_defaults(ctx, ApplyPCAOptions, config)
    shim_write_back_before(ctx, config_file, config, opts)
    result = steps.apply_pca(input_dir, output_dir, output_file, opts)
    shim_write_back_after(config_file, result)


@cli.command('compute-changepoints',
             help='Compute the Model-Free Syllable Changepoints based on the PCA/PCA_Scores')
@click_options(ChangepointsOptions, order=CHANGEPOINTS_ORDER, others=_paths('changepoints'))
@click.pass_context
def compute_changepoints(ctx, input_dir, output_dir, output_file, config_file, **_):
    config = read_config_option(config_file)
    opts = shim_click_defaults(ctx, ChangepointsOptions, config)
    shim_write_back_before(ctx, config_file, config, opts)
    result = steps.compute_changepoints(input_dir, output_dir, output_file, opts)
    shim_write_back_after(config_file, result)


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

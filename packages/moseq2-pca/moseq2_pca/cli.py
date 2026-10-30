"""
CLI for PCA and model-free changepoint analysis.
"""

import os
import click
from os.path import join, exists, expanduser
from moseq2_pca.util import command_with_config, combine_new_config
from moseq2_pca.helpers.wrappers import (train_pca_wrapper, apply_pca_wrapper,
                                         compute_changepoints_wrapper, clip_scores_wrapper)

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass

def common_pca_options(function):
    """
    Decorator function for common Click parameters/dependencies for PCA-related operations.
    
    Args:
    function: Function to add enclosed parameters to as click options.

    Returns:
    function: Updated function including shared parameters.
    """

    function = click.option('--cluster-type', type=click.Choice(['local', 'slurm', 'nodask']),
                  default='local', help='Compute enviornment the command runs in')(function)
    function = click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find extracted h5 files')(function)
    function = click.option('--output-dir', '-o', default=join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store PCA results')(function)
    function = click.option('--config-file', type=click.Path(), default=None, help="Path to configuration file")(function)

    function = click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')(function)
    function = click.option('--h5-mask-path', default='/frames_mask', type=str, help="Path to log-likelihood mask in h5 files")(function)
    function = click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')(function)

    return function


def common_dask_parameters(function):
    """
    Decorator function for common Click parameters for dask.

    Args:
    function: Function to add enclosed parameters to as click options.
    
    Returns:
    function: Updated function including shared parameters.
    """

    function = click.option('--dask-cache-path', '-d', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(),
                            help='Path to spill data to disk for dask')(function)
    function = click.option('--dask-port', default='8787', type=str, help="Port to access dask dashboard")(function)
    function = click.option('-q', '--queue', type=str, default='debug',
                            help="Cluster queue/partition for submitting jobs")(function)
    function = click.option('-n', '--nworkers', type=int, default=1, help="Number of workers")(function)
    function = click.option('-c', '--cores', type=int, default=1, help="Number of cores per worker")(function)
    function = click.option('-p', '--processes', type=int, default=1, help="Number of processes to run on each worker")(
        function)
    function = click.option('-m', '--memory', type=str, default="15GB", help="Total RAM usage per worker")(function)
    function = click.option('-w', '--wall-time', type=str, default="06:00:00", help="Wall time (compute time) for workers")(function)
    function = click.option('--timeout', type=float, default=5,
                            help="Time to wait for workers to initialize before proceeding (minutes)")(function)

    return function


@cli.command(name='train-pca', cls=command_with_config('config_file'), help='Train PCA on all extracted results (h5 files) in input directory')
@common_pca_options
@common_dask_parameters
@click.option('--gaussfilter-space', default=(1.5, 1), type=(float, float), help="x, y sigma for kernel in Spatial filter for data (Gaussian)")
@click.option('--gaussfilter-time', default=0, type=float, help="sigma for temporal filter for data (Gaussian)")
@click.option('--medfilter-space', default=[0], type=int, help="kernel size for median spatial filter", multiple=True)
@click.option('--medfilter-time', default=[0], type=int, help="kernel size for median temporal filter", multiple=True)
@click.option('--missing-data', is_flag=True, default=False, type=bool, help="Use missing data PCA; will be automatically set to True if cable-filter-iters > 1 from the extract step.")
@click.option('--missing-data-iters', default=10, type=int, help="number of missing data PCA iterations")
@click.option('--mask-threshold', default=-16, type=float, help="Threshold for mask (missing data PCA only)")
@click.option('--mask-height-threshold', default=5, type=float, help="Threshold for mask based on floor height")
@click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=120, type=int, help='Max mouse height from floor (mm)')
@click.option('--tailfilter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tailfilter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--use-fft', type=bool, is_flag=True, default=False, help='Use 2D fft')
@click.option('--train-on-subset', default=1, type=float, help="The fraction of the total frames the PCA is trained on; default PCA is trained on all frames")
@click.option('--recon-pcs', type=int, default=10, help='Number of PCs to use for missing data reconstruction')
@click.option('--rank', default=25, type=int, help="Rank for compressed SVD")
@click.option('--output-file', default='pca', type=str, help='Name of h5 file for storing pca results')
@click.option('--local-processes', default=False, type=bool, help='Used with a local cluster. If True: use processes, If False: use threads')
@click.option('--overwrite-pca-train', default=False, type=bool, help='Used to bypass the pca overwrite question. If True: skip question, run automatically')
@click.option('--camera-type', default='k2', type=str, help='specify the camera type (k2 or azure), default is k2')
def train_pca(input_dir, output_dir, output_file, **cli_args):
    # function writes output pca path to config_data
    if cli_args.get('camera_type') == 'azure':
        # check if parameters are set to k2 default, change to azure default
        print('Updating parameters for Azure Kinect camera...')
        if cli_args['gaussfilter_space'] == (1.5, 1):
            cli_args['gaussfilter_space'] = (2.25, 1.5)
        if cli_args['tailfilter_size'] == (9, 9):
            cli_args['tailfilter_size'] = (13, 13)

    config_data = train_pca_wrapper(input_dir, cli_args, output_dir, output_file)
    # write config_data to config_file if there is one
    if cli_args.get('config_file'):
        # combine new config with old config to add output pca path to config.yaml
        combine_new_config(cli_args.get('config_file'), config_data)
    

@cli.command(name='apply-pca', cls=command_with_config('config_file'), help='Compute PCA Scores of extraction data given a pre-trained PCA')
@common_pca_options
@common_dask_parameters
@click.option('--output-file', default='pca_scores', type=str, help='Name of h5 file for storing pca results')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components in h5 file')
@click.option('--pca-file', default=None, type=click.Path(), help='Path to PCA results')
@click.option('--fill-gaps', default=True, type=bool, help='Fill dropped frames with nans')
@click.option('--fps', default=30, type=int, help='Frames per second (frame rate)')
@click.option('--detrend-window', default=0, type=float, help="Length of detrend window (in seconds, 0 for no detrending)")
@click.option('--verbose', '-v', is_flag=True, default=False, help='Print sessions as they are being loaded.')
@click.option('--overwrite-pca-apply', default=False, type=bool, help='Used to bypass the pca overwrite question. If True: skip question, run automatically')
def apply_pca(input_dir, output_dir, output_file, **cli_args):
    # function writes output pc score path to config_data
    config_data, _ = apply_pca_wrapper(input_dir, cli_args, output_dir, output_file)
    # write config_data to config_file if there is one
    if cli_args.get('config_file'):
        # combine new config with old config to add output pc score path to config.yaml
        combine_new_config(cli_args.get('config_file'), config_data)
        

@cli.command('compute-changepoints', cls=command_with_config('config_file'), help='Compute the Model-Free Syllable Changepoints based on the PCA/PCA_Scores')
@common_pca_options
@common_dask_parameters
@click.option('--output-file', default='changepoints', type=str, help='Name of h5 file for storing pca results')
@click.option('--pca-file-components', type=click.Path(), default=None, help="Path to PCA components")
@click.option('--pca-file-scores', type=click.Path(), default=None, help='Path to PCA results')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--neighbors', type=int, default=1, help="Neighbors to use for peak identification")
@click.option('--threshold', type=float, default=.5, help="Peak threshold to use for changepoints")
@click.option('-k', '--klags', type=int, default=6, help="Lag to use for derivative calculation")
@click.option('-s', '--sigma', type=float, default=3.5, help="Half-width of the smoothing window in frames, rounded down to whole frames. The Gaussian inside it has a standard deviation of 1.5 frames. Also sets how many frames are trimmed from each end of the changepoint curve (6 x sigma).")
@click.option('-d', '--dims', type=int, default=300, help="Number of random projections to use")
@click.option('--fps', default=30, type=int, help="Frames per second (frame rate)")
@click.option('--verbose', '-v', is_flag=True, default=False, help="Print sessions as they are being loaded.")
def compute_changepoints(input_dir, output_dir, output_file, **cli_args):
    # function writes output changepoint path to config_data
    config_data = compute_changepoints_wrapper(input_dir, cli_args, output_dir, output_file)
    # write config_data to config_file if there is one
    if cli_args.get('config_file'):
        # combine new config with old config to add output pc score path to config.yaml
        combine_new_config(cli_args.get('config_file'), config_data)
    

@cli.command('clip-scores',  help='Clip specified number of frames from PCA scores at the beginning or end')
@click.argument('pca_file', type=click.Path(exists=True, resolve_path=True))
@click.argument('clip_samples', type=int)
@click.option('--from-end', type=bool, is_flag=True, default=False, help="if true clip from end rather than beginning")
def clip_scores(pca_file, clip_samples, from_end):
    clip_scores_wrapper(pca_file, clip_samples, from_end)

if __name__ == '__main__':
    cli()
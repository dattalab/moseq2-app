"""The settings of each PCA step, declared once.

One class per command. Each field carries its default, its command-line flags,
its click type and its help text; ``cli.py`` generates the commands' options
from these classes, and ``steps.py`` takes an instance. The paths a step works
on (``--input-dir``, ``--output-dir``, ``--output-file``, ``--pca-file``,
``--pca-file-scores``) and ``--config-file`` are not settings: they are the
steps' arguments and plain options in ``cli.py``.

The three steps' classes read ``config.yaml``'s ``pca`` section
(``PCA_STAGE``), whose height range, chunk size and compute resources are PCA's
own, apart from extraction's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import click

from moseq2_common.options import Options, option


@dataclass(frozen=True, kw_only=True)
class PCAInputOptions(Options):
    """Where the extracted frames are and how they are read."""

    SECTION = "pca"

    cluster_type: str = option('local', '--cluster-type', type=click.Choice(['local', 'slurm', 'nodask']),
                               help='Compute enviornment the command runs in')
    chunk_size: int = option(4000, '--chunk-size', type=int, help='Number of frames per chunk')
    h5_path: str = option('/frames', '--h5-path', type=str, help='Path to data in h5 files')
    h5_mask_path: str = option('/frames_mask', '--h5-mask-path', type=str,
                               help="Path to log-likelihood mask in h5 files")


@dataclass(frozen=True, kw_only=True)
class DaskOptions(Options):
    """The dask cluster a PCA step runs on."""

    SECTION = "pca"

    dask_cache_path: str = option(os.path.join(os.getcwd(), '_pca'), '--dask-cache-path', '-d',
                                  type=click.Path(), help='Path to spill data to disk for dask')
    dask_port: str = option('8787', '--dask-port', type=str, help="Port to access dask dashboard")
    queue: str = option('debug', '-q', '--queue', type=str, help="Cluster queue/partition for submitting jobs")
    nworkers: int = option(1, '-n', '--nworkers', type=int, help="Number of workers")
    cores: int = option(1, '-c', '--cores', type=int, help="Number of cores per worker")
    processes: int = option(1, '-p', '--processes', type=int, help="Number of processes to run on each worker")
    memory: str = option("15GB", '-m', '--memory', type=str, help="Total RAM usage per worker")
    wall_time: str = option("06:00:00", '-w', '--wall-time', type=str, help="Wall time (compute time) for workers")
    timeout: float = option(5, '--timeout', type=float,
                            help="Time to wait for workers to initialize before proceeding (minutes)")


@dataclass(frozen=True, kw_only=True)
class TrainPCAOptions(PCAInputOptions, DaskOptions):
    """``train-pca``."""

    gaussfilter_space: tuple = option((1.5, 1), '--gaussfilter-space', type=(float, float),
                                      help="x, y sigma for kernel in Spatial filter for data (Gaussian)")
    gaussfilter_time: float = option(0, '--gaussfilter-time', type=float,
                                     help="sigma for temporal filter for data (Gaussian)")
    medfilter_space: tuple = option((0,), '--medfilter-space', type=int, multiple=True,
                                    help="kernel size for median spatial filter")
    medfilter_time: tuple = option((0,), '--medfilter-time', type=int, multiple=True,
                                   help="kernel size for median temporal filter")
    missing_data: bool = option(False, '--missing-data', is_flag=True, type=bool,
                                help="Use missing data PCA")
    missing_data_iters: int = option(10, '--missing-data-iters', type=int,
                                     help="number of missing data PCA iterations")
    mask_threshold: float = option(-16, '--mask-threshold', type=float,
                                   help="Threshold for mask (missing data PCA only)")
    mask_height_threshold: float = option(5, '--mask-height-threshold', type=float,
                                          help="Threshold for mask based on floor height")
    min_height: int = option(10, '--min-height', type=int, help='Min mouse height from floor (mm)')
    max_height: int = option(120, '--max-height', type=int, help='Max mouse height from floor (mm)')
    tailfilter_size: tuple = option((9, 9), '--tailfilter-size', type=(int, int), help='Tail filter size')
    tailfilter_shape: str = option('ellipse', '--tailfilter-shape', type=str, help='Tail filter shape')
    use_fft: bool = option(False, '--use-fft', type=bool, is_flag=True, help='Use 2D fft')
    train_on_subset: float = option(1, '--train-on-subset', type=float,
                                    help="The fraction of the total frames the PCA is trained on; "
                                         "default PCA is trained on all frames")
    recon_pcs: int = option(10, '--recon-pcs', type=int, help='Number of PCs to use for missing data reconstruction')
    rank: int = option(25, '--rank', type=int, help="Rank for compressed SVD")
    local_processes: bool = option(False, '--local-processes', type=bool,
                                   help='Used with a local cluster. If True: use processes, If False: use threads')
    overwrite_pca_train: bool = option(False, '--overwrite-pca-train', type=bool,
                                       help='Used to bypass the pca overwrite question. If True: skip question, '
                                            'run automatically')


@dataclass(frozen=True, kw_only=True)
class ApplyPCAOptions(PCAInputOptions, DaskOptions):
    """``apply-pca``."""

    pca_path: str = option('/components', '--pca-path', type=str, help='Path to pca components in h5 file')
    verbose: bool = option(False, '--verbose', '-v', is_flag=True, setting=False,
                           help='Print sessions as they are being loaded.')
    overwrite_pca_apply: bool = option(False, '--overwrite-pca-apply', type=bool,
                                       help='Used to bypass the pca overwrite question. If True: skip question, '
                                            'run automatically')


@dataclass(frozen=True, kw_only=True)
class ChangepointsOptions(PCAInputOptions, DaskOptions):
    """``compute-changepoints``."""

    pca_path: str = option('/components', '--pca-path', type=str, help='Path to pca components')
    neighbors: int = option(1, '--neighbors', type=int, help="Neighbors to use for peak identification")
    threshold: float = option(.5, '--threshold', type=float, help="Peak threshold to use for changepoints")
    klags: int = option(6, '-k', '--klags', type=int, help="Lag to use for derivative calculation")
    sigma: float = option(3.5, '-s', '--sigma', type=float,
                          help="Half-width of the smoothing window in frames, rounded down to whole frames. "
                               "The Gaussian inside it has a standard deviation of 1.5 frames. Also sets how "
                               "many frames are trimmed from each end of the changepoint curve (6 x sigma).")
    # -d is also --dask-cache-path's short flag, as in 1.5.0; click gives it to --dims
    dims: int = option(300, '-d', '--dims', type=int, help="Number of random projections to use")
    verbose: bool = option(False, '--verbose', '-v', is_flag=True, setting=False,
                           help="Print sessions as they are being loaded.")


@dataclass(frozen=True, kw_only=True)
class ClipScoresOptions(Options):
    """``clip-scores``."""

    from_end: bool = option(False, '--from-end', type=bool, is_flag=True,
                            help="if true clip from end rather than beginning")


#: The classes of the PCA steps, whose fields config.yaml's ``pca`` section may set.
PCA_STAGE = (TrainPCAOptions, ApplyPCAOptions, ChangepointsOptions)

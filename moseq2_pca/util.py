"""
Utility and helper functions for finding and reading files, filtering operations, Dask initialization, and changepoint helper functions.
"""

import os
import cv2
import h5py
import time
import dask
import click
from click.core import ParameterSource
import psutil
import warnings
import platform
import subprocess
import numpy as np
import scipy.signal
from glob import glob
from copy import deepcopy
from moseq2_pca._yaml import read_yaml as _read_yaml, write_yaml
from tqdm.auto import tqdm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from os.path import join, exists, abspath, expanduser


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):
    """
    Helper function to assign variables from a config file. 
    Hierachy of CLI prameters: params from cli options > params from config_file > default params
    
    Args:
    config_file_param_name (str): parameter name to update with config file variable.

    Returns:
    custom_command_class (click.Command): updated Click Command containing parameters from inputted config file.
    """

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]

            if config_file is not None:
                # read params from config_file
                config_data = read_yaml(config_file)

                # set config_data['output_file'] ['output_dir'] ['input_dir'] to None to avoid overwriting previous files
                config_data['input_dir'] = None
                config_data['output_dir'] = None
                config_data['output_file'] = None

                for param, value in ctx.params.items():
                    # set params to the params in config file when the param is not none
                    if param in config_data and config_data[param]:
                        if type(value) is tuple and type(config_data[param]) is int:
                            ctx.params[param] = tuple([config_data[param]])
                        elif type(value) is tuple:
                            ctx.params[param] = tuple(config_data[param])
                        else:
                            ctx.params[param] = config_data[param]

                        # a value the user typed on the command line outranks the
                        # config file. click records where each value came from, which
                        # answers that directly; comparing the resolved value against
                        # the declared default cannot, because the two coincide
                        # whenever someone passes an option at its default value.
                        if ctx.get_parameter_source(param) is ParameterSource.COMMANDLINE:
                            ctx.params[param] = value

                # removed flags
                flag_list = ['missing_data', 'use_fft', 'verbose', 'from_end']
                combined = {k:v for k,v in ctx.params.items() if k not in flag_list}
                # combine params with config_params
                config_data = {**config_data, **combined}
                # write parameters to config_file
                write_yaml(config_data, config_file)
            return super(custom_command_class, self).invoke(ctx)

    return custom_command_class


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """
    Recursively find h5 files, along with yaml files with the same basename

    Args:
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for
    yaml_string (str): string for filename formatting when saving data

    Returns:
    h5s (list): list of found h5 files
    dicts (list): list of found metadata files
    yamls (list): list of found yaml files
    """

    if not ext.startswith('.'):
        ext = '.' + ext

    def has_frames(f):
        try:
            with h5py.File(f, 'r') as h5f:
                return 'frames' in h5f
        except OSError:
            warnings.warn(f'Error reading {f}, skipping...')
            return False

    h5s = glob(join(abspath(root_dir), '**', f'*{ext}'), recursive=True)
    h5s = filter(lambda f: exists(yaml_string.format(f.replace(ext, ''))), h5s)
    h5s = list(filter(has_frames, h5s))
    yamls = list(map(lambda f: yaml_string.format(f.replace(ext, '')), h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def gauss_smooth(signal, win_length=None, sig=1.5, kernel=None):
    """
    Perform Gaussian Smoothing on a 1D signal.

    Args:
    signal (1d numpy array): signal to perform smoothing
    win_length (int): window size for gaussian kernel filter
    sig (float): variance of 1d gaussian kernel.
    kernel (tuple): kernel size to use for smoothing

    Returns:
    result (1d numpy array): smoothed signal
    """
    if kernel is None:
        kernel = gaussian_kernel1d(n=win_length, sig=sig)

    result = scipy.signal.convolve(signal, kernel, mode='same', method='direct')

    return result


def gaussian_kernel1d(n=None, sig=3):
    """
    Get 1D gaussian kernel.

    Args:
    n (int): window size.
    sig (int): variance of kernel to use.

    Returns:
    kernel (1d array): 1D numpy kernel.
    """

    if n is None:
        n = np.ceil(sig * 4)

    points = np.arange(-n, n)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 medfilter_time=None, gaussfilter_time=None, detrend_time=None,
                 tailfilter=None, tail_threshold=5):
    """
    Filter spatial/temporal noise from frames using Median and Gaussian filters,
    given kernel sizes for each respective requested filter.

    Args:
    frames (numpy.ndarray): frames to filter.
    medfilter_space (list): median spatial filter kernel.
    gaussfilter_space (list): gaussian spatial filter kernel.
    medfilter_time (list): median temporal filter.
    gaussfilter_time (list): gaussian temporal filter.
    detrend_time (int): number of frames to lag for.
    tailfilter (int): size of tail-filter kernel.
    tail_threshold (int): threshold value to use for tail filtering

    Returns:
    out (numpy.ndarray): filtered frames.
    """

    out = np.copy(frames)

    if tailfilter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(out[i], cv2.MORPH_OPEN, tailfilter) > tail_threshold
            out[i] = out[i] * mask.astype(frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                out[i] = cv2.medianBlur(out[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(out[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    if medfilter_time is not None and np.all(np.array(medfilter_time) > 0):
        for idx, i in np.ndenumerate(frames[0]):
            for medfilt in medfilter_time:
                out[:, idx[0], idx[1]] = \
                    scipy.signal.medfilt(out[:, idx[0], idx[1]], medfilt)

    if gaussfilter_time is not None and gaussfilter_time > 0:
        kernel = gaussian_kernel1d(sig=gaussfilter_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = \
                np.convolve(out[:, idx[0], idx[1]], kernel, mode='same')

    if detrend_time is not None and detrend_time > 0:
        kernel = gaussian_kernel1d(sig=detrend_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = \
                out[:, idx[0], idx[1]] - gauss_smooth(out[:, idx[0], idx[1]], kernel=kernel)

    return out


def select_strel(string='e', size=(10, 10)):
    """
    Select Structuring Element Shape. Accepts shapes ('ellipse', 'rectangle'), if neither
    are given then 'ellipse' is used.

    Args:
    string (str): e for Ellipse, r for Rectangle
    size (tuple): size of StructuringElement

    Returns:
    strel (cv2.StructuringElement): StructuringElement with specified size.
    """
    if not isinstance(size, tuple):
        size = tuple(size)

    if string is None or 'none' in string or np.all(np.array(size) == 0) or len(string) == 0:
        strel = None
    elif string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return strel


def insert_nans(timestamps, data, fps=30):
    """
    Fill NaN values with 0 in given 1D timestamps array. Used to handle dropped frames from the video acquisition.

    Args:
    timestamps (numpy.array): timestamp values
    data (numpy.array): additional data to fill with NaN values - can be PC scores
    fps (int): frames per second

    Returns:
    filled_data (numpy.array): filled missing timestamp values.
    data_idx (numpy.array): indices of inserted 0s
    filled_timestamps (numpy.array): filled timestamp-strs
    """

    df_timestamps = np.diff(np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))

    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)

    if filled_data.ndim == 1:
        isvec = True
        filled_data = filled_data[:, None]
    else:
        isvec = False

    nframes, nfeatures = filled_data.shape

    for idx in fill_idx[::-1]:
        if idx < len(missing_frames): # ensures ninserts value remains an int
            ninserts = int(missing_frames[idx] - 1)
            data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
            insert_timestamps = timestamps[idx - 1] + \
                np.cumsum(np.ones(ninserts,) * 1.0 / fps)
            filled_data = np.insert(filled_data, idx,
                                    np.ones((ninserts, nfeatures)) * np.nan, axis=0)
            filled_timestamps = np.insert(
                filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def read_yaml(yaml_file):
    """
    Read yaml file and return dictionary representation of file contents.

    Args:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml file contents
    """

    try:
        return_dict = _read_yaml(yaml_file)
    except IOError:
        return_dict = {}

    return return_dict


def check_timestamps(h5s):
    """
    Helper function to determine whether timestamps and/or metadata is missing from
    extracted files. Function will emit a warning if either pieces of data are missing.

    Args:
    h5s (list): List of paths to all extracted h5 files.
    """

    for h5 in h5s:
        try:
            h5_timestamp_path = get_timestamp_path(h5)
        except:
            warnings.warn(f'Autoload timestamps for session {h5} failed.')
            h5_timestamp_path = None
        try:
            h5_metadata_path = get_metadata_path(h5)
        except:
            warnings.warn(f'Autoload metadata for session {h5} failed.')
            h5_metadata_path = None

        if h5_timestamp_path is None:
            warnings.warn(f'Could not located timestamps in {h5}. \
                          This may cause issues if PCA has been trained on missing data.')
        if h5_metadata_path is None:
            warnings.warn(f'Could not located metadata in {h5}. \
                          This may cause issues if PCA has been trained on missing data.')


def get_timestamp_path(h5file):
    """
    Return path within h5 file that contains the kinect timestamps

    Args:
    h5file (str): path to h5 file.

    Returns:
    (str): path to metadata timestamps within h5 file
    """

    with h5py.File(h5file, 'r') as f:
        if '/timestamps' in f:
            return '/timestamps'
        elif '/metadata/timestamps' in f:
            return '/metadata/timestamps'
        else:
            raise KeyError('timestamp key not found')


def get_metadata_path(h5file):
    """
    Return path within h5 file that contains the kinect extraction metadata.

    Args:
    h5file (str): path to h5 file.

    Returns:
    (str): path to acquistion metadata within h5 file.
    """

    with h5py.File(h5file, 'r') as f:
        if '/metadata/acquisition' in f:
            return '/metadata/acquisition'
        elif '/metadata/extraction' in f:
            return '/metadata/extraction'
        else:
            raise KeyError('acquisition metadata not found')


def h5_to_dict(h5file, path):
    """
    Read all contents from h5 and returns them in a nested dict object.

    Args:
    h5file (str): path to h5 file
    path (str): path to group within h5 file

    Returns:
    ans (dict): dictionary of all h5 group contents
    """

    ans = {}

    if type(h5file) is str:
        with h5py.File(h5file, 'r') as f:
            ans = h5_to_dict(f, path)
            return ans

    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = _read_dataset(item)
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans


def _read_dataset(dataset):
    """Read one dataset, decoding variable-length UTF-8 strings to `str`.

    h5py 3 returns the stored bytes for a string dataset and supplies
    `asstr()` to decode them; h5py 2 decodes variable-length UTF-8 on read and
    has no `asstr()`. Decoding here keeps a session's uuid, subject name and
    acquisition timestamps coming back as `str` on either, which matters
    because callers compare those against strings read from YAML indexes and
    use them as dictionary keys, where `b'...'` and `'...'` do not match.

    Only variable-length UTF-8 is decoded. Fixed-length and ASCII string
    datasets come back as `bytes` under h5py 2 as well, so decoding those would
    introduce a difference rather than remove one.

    A dataset with a null dataspace, which has `shape` of None and is how an
    extraction records a parameter left unset, is read directly. It holds no
    value to decode, and `asstr()` raises on one.
    """
    info = h5py.check_string_dtype(dataset.dtype)
    if (info is not None and info.length is None and info.encoding == 'utf-8'
            and dataset.shape is not None and hasattr(dataset, 'asstr')):
        return dataset.asstr()[()]
    return dataset[()]


def set_dask_config(memory={'target': 0.85, 'spill': False, 'pause': False, 'terminate': 0.95}):
    """
    Set initial dask configuration parameters

    Args:
    memory (dict): dictionary containing default dask configuration variables to ensure safe amount of resource usage.
    """

    memory = {f'distributed.worker.memory.{k}': v for k, v in memory.items()}
    dask.config.set(memory)
    dask.config.set({'optimization.fuse.ave-width': 5})


def get_env_cpu_and_mem():
    """
    Read current system environment and return the amount of available memory and CPUs to allocate to the created cluster.

    Returns:
    mem (float): Optimal number of memory (in bytes) to allocate to initialized dask cluster
    cpu (int): Optimal number of CPUs to allocate to dask
    """

    is_slurm = os.environ.get('SLURM_JOBID', False)

    if is_slurm:
        click.echo('Detected slurm environment, using "sacct" to detect cpu and memory requirements')
        cmd = f'sacct -j {is_slurm} --format AllocCPUS,ReqMem -X -n -p'
        output = subprocess.check_output(cmd.split(' '))
        output = output.decode('utf-8').strip().split('|')
        cpu, mem, _ = output
        cpu = max(1, int(cpu)-1)

        if 'G' in mem:
            # account for additional processes that needs memory
            mem = float(mem[:mem.index('G')]) * 1e9 * 0.8
        elif 'M' in mem:
            # account for additional processes that needs memory
            mem = float(mem[:mem.index('M')]) * 1e6 * 0.8
    else:
        mem = psutil.virtual_memory().available * 0.8
        cpu = max(1, psutil.cpu_count() - 1)

    return mem, cpu


def initialize_dask(nworkers=50, processes=1, memory='4GB', cores=1,
                    wall_time='01:00:00', queue='debug', local_processes=False,
                    cluster_type='local', timeout=10,
                    cache_path=expanduser('~/moseq2_pca'),
                    dashboard_port='8787', data_size=None, **kwargs):
    """
    Initialize dask client, cluster, workers, etc.

    Args:
    nworkers (int): number of dask workers to initialize
    processes (int): number of processes per worker
    memory (str): amount of memory to allocate to dask cluster
    cores (int): number of cores to use.
    wall_time (str): amount of time to allow program to run
    queue (str): logging mode
    local_processes (bool): flag to use processes or threads when using a local cluster
    cluster_type (str): indicate what cluster to use (local or slurm)
    timeout (int): how many minutes to wait for workers to initialize
    cache_path (str or Pathlike): path to store cached data
    dashboard_port (str): port number to find dask statistics
    data_size (float): size of the dask array in number of bytes.
    kwargs: extra keyward arguments

    Returns:
    client (dask Client): initialized Client
    cluster (dask Cluster): initialized Cluster
    workers (dask Workers): intialized workers
    """

    click.echo(f'Access dask dashboard at http://localhost:{dashboard_port}')

    if cluster_type == 'local':

        max_mem, max_cpu = get_env_cpu_and_mem()
        overhead = 0.8e9  # memory overhead for each worker; approximate
        
        # allocating 0.4 of the maximum memory to account for overhead per worker
        allowed = max_mem * 0.4 
        max_workers = allowed // overhead

        # set number of workers to maximum workers, or total number of CPUs
        if nworkers > max_workers:
            click.echo(f'Reducing number of workers to {min(max_workers, max_cpu)} to account for worker base memory and the number of CPUs')
        nworkers = int(min(max(1, nworkers), max_workers, max_cpu))

        # compute mem limit per worker
        try:
            mem_limit = max(1, max_mem / nworkers)
        except:
            mem_limit = 1

        # display some diagnostic info
        if data_size is not None:
            click.echo(f'Dataset size: {round(data_size / 1e9, 2)}GB')
        click.echo(f'Setting number of workers to: {nworkers}')
        click.echo(f'Overriding memory per worker to {round(mem_limit / 1e9, 2)}GB')

        client = Client(processes=local_processes,
                        threads_per_worker=1,
                        memory_limit=mem_limit,
                        n_workers=nworkers,
                        dashboard_address=dashboard_port,
                        local_directory=cache_path,
                        **kwargs)
        cluster = client.cluster

    elif cluster_type == 'slurm':

        cluster = SLURMCluster(processes=processes,
                               n_workers=nworkers,
                               cores=cores,
                               memory=memory,
                               queue=queue,
                               walltime=wall_time,
                               local_directory=cache_path,
                               scheduler_options={'dashboard_address': dashboard_port},
                               **kwargs)
        client = Client(cluster)
    else:
        raise NotImplementedError('Specified cluster not supported. Supported types are: "slurm", "local"')

    if client is not None:

        client_info = client.scheduler_info()
        if 'services' in client_info.keys() and 'bokeh' in client_info['services'].keys():
            ip = client_info['address'].split('://')[1].split(':')[0]
            port = client_info['services']['bokeh']
            hostname = platform.node()
            click.echo(f'Web UI served at {ip}:{port} (if port forwarding use internal IP not localhost)')
            click.echo(f'Tunnel command:\n ssh -NL {port}:{ip}:{port} {hostname}')
            click.echo(f'Tunnel command (gcloud):\n gcloud compute ssh {hostname} -- -NL {port}:{ip}:{port}')

    if cluster_type == 'slurm':

        active_workers = len(client.scheduler_info()['workers'])
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pbar = tqdm(total=nworkers, desc="Intializing workers")

            elapsed_time = (time.time() - start_time) / 60

            while active_workers < nworkers and elapsed_time < timeout:
                tmp = len(client.scheduler_info()['workers'])
                if tmp - active_workers > 0:
                    pbar.update(tmp - active_workers)
                active_workers = tmp
                time.sleep(1)
                elapsed_time = (time.time() - start_time) / 60

            pbar.close()

    workers = cluster.workers

    return client, cluster, workers


def close_dask(client, cluster, timeout):
    """
    Shut down the Dask client and cluster, and dump all cache data.

    Args:
    client (Dask Client): Client object
    cluster (dask Cluster): initialized Cluster
    timeout (int): Time to wait for client to close gracefully (minutes)

    Returns:
    """

    if client is not None:
        try:
            client.close(timeout=timeout)
            cluster.close(timeout=timeout)
        except Exception as e:
            print('Error:', e)
            print('Could not shutdown dask client')


def get_rps(frames, rps=600, normalize=True):
    """
    Get random projections of frames.

    Args:
    frames (numpy.array): Frames to get dimensions from.
    rps (int): Number of random projections.
    normalize (bool): indicates whether to normalize the random projections.

    Returns:
    rproj (2D or 3D numpy array): Computed random projections with same shape as frames
    """

    if frames.ndim == 3:
        use_frames = frames.reshape(-1, np.prod(frames.shape[1:]))
    elif frames.ndim == 2:
        use_frames = frames

    rproj = use_frames.dot(np.random.randn(use_frames.shape[1], rps).astype('float32'))

    if normalize:
        rproj = scipy.stats.zscore(scipy.stats.zscore(rproj).T)

    return rproj


def get_changepoints(scores, k=5, sigma=3, peak_height=.5, peak_neighbors=1,
                     baseline=True, timestamps=None):
    """
    Compute changepoints and its corresponding distribution. Changepoints describe
    the magnitude of frame-to-frame changes of mouse pose.

    Args:
    scores (numpy.ndarray): nframes * rows * columns
    k (int): klags - Lag to use for derivative calculation.
    sigma (int): Standard deviation of gaussian smoothing filter.
    peak_height (float): user-defined peak Changepoint length.
    peak_neighbors (int): number of peaks in the CP curve.
    baseline (bool): normalize data.
    timestamps (numpy.array): loaded timestamps.

    Returns:
    cps (numpy.ndarray): array of changepoint values
    normed_df (numpy.array): array of values for bar plot
    """

    if type(k) is not int:
        k = int(k)

    if type(peak_neighbors) is not int:
        peak_neighbors = int(peak_neighbors)

    normed_df = deepcopy(scores)
    nanidx = np.isnan(normed_df)
    normed_df[nanidx] = 0

    if sigma is not None and sigma > 0:
        for i in range(scores.shape[0]):
            normed_df[i, :] = gauss_smooth(normed_df[i, :], sigma)

    normed_df[:, k // 2:-k // 2] = (normed_df[:, k:] - normed_df[:, :-k])**2

    normed_df[nanidx] = np.nan
    normed_df[:, :int(6 * sigma)] = np.nan
    normed_df[:, -int(6 * sigma):] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normed_df = np.nanmean(normed_df, axis=0)

        if baseline:
            normed_df -= np.nanmin(normed_df)

        if timestamps is not None:
            normed_df, _, _ = insert_nans(
                timestamps, normed_df, fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

        normed_df = np.squeeze(normed_df)
        cps = scipy.signal.argrelextrema(
            normed_df, np.greater, order=peak_neighbors)[0]
        cps = cps[np.argwhere(normed_df[cps] > peak_height)]

    return cps, normed_df


def combine_new_config(config_file, config_data):
    """
    Read config file and combine new config params with it

    Args:
        config_file (str): path to config.yaml
        config_data (dict): dictionary of config data
    """
    # open the config file
    temp_config = _read_yaml(config_file)
    # combining config data with the existing config file
    config_data = {**temp_config, **config_data}
    # ensure output_file and output_dir are not in config_data or reusing config for extraction will fail
    config_data = {k:v for k, v in config_data.items() if k not in ('output_dir', 'output_file')}
    write_yaml(config_data, config_file)
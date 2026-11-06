"""
Utility and helper functions for finding and reading files, filtering operations, Dask initialization, and changepoint helper functions.
"""

import os
import sys
import asyncio
import h5py
import time
import dask
import logging
import click
import psutil
import warnings
import platform
import shutil
import subprocess
import numpy as np
import scipy.signal
from glob import glob
from copy import deepcopy
from contextlib import contextmanager
from moseq2_common.bookkeeping import recording_order
from moseq2_common.frames import clean_frames as clean_frames_spatial, odd_kernel
from moseq2_common.h5 import extraction_labels
from moseq2_common.records import StepInputError
from moseq2_common.timestamps import frame_steps, timestamps_path
from moseq2_common.yaml import write_yaml
from tqdm.auto import tqdm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from os.path import join, exists, abspath, expanduser


def recursive_find_h5s(root_dir=os.getcwd(), ext='.h5'):
    """
    Recursively find the extractions under ``root_dir``: the ``.h5`` files holding frames and a session uuid,
    listed in the order the sessions were recorded (``moseq2_common.bookkeeping.recording_order``), as a
    project lists them.

    Args:
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for

    Returns:
    h5s (list): list of found h5 files
    dicts (list): each extraction's ``uuid`` and acquisition ``metadata``, read from its ``.h5``
    sources (list): the files each session's uuid is read from: the ``.h5`` files themselves
    """

    if not ext.startswith('.'):
        ext = '.' + ext

    def extraction(f):
        try:
            with h5py.File(f, 'r') as h5f:
                if 'frames' not in h5f or 'metadata/uuid' not in h5f:
                    return None
            uuid, metadata = extraction_labels(f)
        except OSError:
            warnings.warn(f'Error reading {f}, skipping...')
            return None
        return {'uuid': uuid, 'metadata': metadata}

    found = [(h5, extraction(h5)) for h5 in glob(join(abspath(root_dir), '**', f'*{ext}'), recursive=True)]
    found = sorted(((h5, info) for h5, info in found if info is not None),
                   key=lambda pair: recording_order(pair[1]['metadata'], pair[0]))
    h5s = [h5 for h5, _ in found]
    dicts = [info for _, info in found]

    return h5s, dicts, list(h5s)


def find_extractions(sessions):
    """
    The extractions a PCA step reads, as ``recursive_find_h5s`` returns them.

    Args:
    sessions (str or list): a folder to search (``recursive_find_h5s``), or the sessions' ``.h5`` files in the
        order to read them (a project's, from ``moseq2_common.bookkeeping.session_index``)

    Returns:
    h5s (list), dicts (list), sources (list): as ``recursive_find_h5s``
    """
    if isinstance(sessions, (str, os.PathLike)):
        return recursive_find_h5s(sessions)
    h5s = [str(h5) for h5 in sessions]
    dicts = []
    for h5 in h5s:
        uuid, metadata = extraction_labels(h5)
        dicts.append({'uuid': uuid, 'metadata': metadata})
    return h5s, dicts, list(h5s)


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
    n (int): half-width of the window in samples, rounded down to a whole sample.
    sig (int): variance of kernel to use.

    Returns:
    kernel (1d array): 1D numpy kernel, centred on zero.
    """

    if n is None:
        n = np.ceil(sig * 4)
    n = int(n)

    points = np.arange(-n, n + 1)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 medfilter_time=None, gaussfilter_time=None, detrend_time=None,
                 tailfilter=None, tail_threshold=5):
    """
    Filter spatial/temporal noise from frames using Median and Gaussian filters,
    given kernel sizes for each respective requested filter. The spatial filters
    are moseq2_common.frames.clean_frames; the temporal filters follow them.
    An even median kernel, spatial or temporal, rounds up to the next odd size
    (moseq2_common.frames.odd_kernel).

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

    out = clean_frames_spatial(frames, medfilter_space=medfilter_space,
                               gaussfilter_space=gaussfilter_space,
                               tail_filter=tailfilter, tail_threshold=tail_threshold)

    if medfilter_time is not None and np.all(np.array(medfilter_time) > 0):
        medfilter_time = [odd_kernel(medfilt) for medfilt in medfilter_time]
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


#: The whole frames per second ``frame_rate_from_timestamps`` accepts. Depth
#: cameras run at 5 to 90 frames per second. A file that records its timestamps'
#: unit is held to 1 to 1,000. A file from before the unit was recorded is read as
#: milliseconds, and an Azure .mkv extracted then stored seconds, so its rate
#: reads 1000 times too high (about 30,000); those files are held to 1 to
#: 1,000,000, and their dropped frames are still placed correctly, because
#: insert_nans takes the frame period from the same timestamps.
FRAME_RATE_BOUNDS = (1, 1_000)
FRAME_RATE_BOUNDS_UNIT_NOT_RECORDED = (1, 1_000_000)


def frame_rate_from_timestamps(timestamps, session=None, unit=None):
    """
    The whole frames per second a session's timestamps imply, from the median
    gap between them, which is the frame period insert_nans places dropped frames
    with. Stops with the values involved when the timestamps cannot give one.

    Args:
    timestamps (numpy.array): the session's timestamps, in seconds.
    session (str): the session's uuid and file, for the error message.
    unit (str): the unit the session's file records for its timestamps, or None
        when it records none.

    Returns:
    fps (int): frames per second, rounded to a whole number.
    """
    timestamps = np.asarray(timestamps, dtype='float64')
    diffs = np.diff(timestamps)
    with np.errstate(divide='ignore', invalid='ignore'):
        period = np.median(diffs) if len(diffs) > 0 else np.nan
        rate = 1 / period
    low, high = FRAME_RATE_BOUNDS if unit is not None else FRAME_RATE_BOUNDS_UNIT_NOT_RECORDED
    if np.isfinite(rate) and low <= np.round(rate) <= high:
        return np.round(rate).astype('int')

    if len(diffs) == 0:
        cause = f'there are only {len(timestamps)} timestamps, and a frame rate needs at least two.'
    elif not np.isfinite(timestamps).all():
        cause = 'some timestamps are NaN or infinite.'
    elif np.all(diffs == 0):
        cause = 'the timestamps are all equal, so no time passes between frames.'
    elif period < 0:
        cause = 'the timestamps decrease, so they are out of order.'
    elif period == 0:
        cause = 'at least half of the frames have the same timestamp as the frame before.'
    elif rate < low:
        cause = (
            'the timestamps are about 1000 times too large, which is what timestamps in '
            'milliseconds look like where seconds are expected (at extraction, a depth_ts.txt '
            'in microseconds rather than milliseconds), or the recording has very long gaps.'
        )
    elif unit is not None:
        cause = (
            f'the timestamps are far smaller than a camera\'s frame interval for the unit the '
            f'file records ({unit}); timestamps in seconds recorded as milliseconds look like this.'
        )
    else:
        cause = 'the timestamps are far smaller than a camera\'s frame interval, so their unit is not the one expected.'
    name = f'session {session}' if session is not None else 'this session'
    first = ', '.join(f'{d:g}' for d in diffs[:5])
    raise ValueError(
        f'Cannot work out the frame rate of {name} from its timestamps, so dropped frames '
        f'cannot be placed. They give {rate:g} frames per second (rounded to {np.round(rate):g}); '
        f'a frame rate between {low:,} and {high:,} is needed. The first timestamp differences '
        f'are [{first}] seconds. Likely cause: {cause} Check the timestamps this session '
        f'was extracted from (the recording\'s depth_ts.txt, one line per frame with the time '
        f'in milliseconds first; its timestamps.csv, in seconds; or, for an .mkv recording '
        f'with neither, the frame times stored in the .mkv) and extract the session again.'
    )


def insert_nans(timestamps, data, fps=None):
    """
    Put an empty (NaN) row in data wherever the camera dropped frames, so that
    every row is one frame step.

    The frame period is the median gap between timestamps, which the drops
    themselves do not move, or 1 / fps when fps is given. A gap of n periods,
    rounded to the nearest whole number, holds n - 1 dropped frames.

    Args:
    timestamps (numpy.array): each row's time, in seconds
    data (numpy.array): rows to fill, one per timestamp - can be PC scores
    fps (float): frames per second to place frames with; by default the
        timestamps' own, from the median gap

    Returns:
    filled_data (numpy.array): data with a NaN row for every dropped frame
    data_idx (numpy.array): each row's index into data, NaN for an inserted row
    filled_timestamps (numpy.array): timestamps with one for every inserted row
    """

    timestamps = np.asarray(timestamps)
    gaps = np.diff(timestamps)
    if fps is not None:
        period = 1.0 / fps
    elif len(gaps) > 0:
        period = np.median(gaps)
    else:
        period = np.nan

    if len(gaps) > 0 and not (np.isfinite(period) and period > 0):
        raise ValueError(
            f'Cannot place dropped frames: the frame period from these timestamps is {period:g} '
            f'seconds, and a positive period is needed. The first timestamp differences are '
            f'[{", ".join(f"{d:g}" for d in gaps[:5])}] seconds. Check the timestamps with '
            f'frame_rate_from_timestamps, which says what is wrong with them.'
        )

    steps = frame_steps(timestamps, period)
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)

    if filled_data.ndim == 1:
        isvec = True
        filled_data = filled_data[:, None]
    else:
        isvec = False

    nframes, nfeatures = filled_data.shape

    # gap i lies between rows i and i + 1; insert from the end so earlier positions hold
    for gap in np.flatnonzero(steps > 1)[::-1]:
        ninserts = int(steps[gap]) - 1
        idx = gap + 1
        data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
        insert_timestamps = timestamps[gap] + np.arange(1, ninserts + 1) * period
        filled_data = np.insert(filled_data, idx,
                                np.ones((ninserts, nfeatures)) * np.nan, axis=0)
        filled_timestamps = np.insert(
            filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def _frame_shapes(h5s, dicts, h5_path='/frames'):
    """Each session's frame size, UUID and file, for the sessions that have frames at h5_path."""
    shapes = []
    for h5, meta in zip(h5s, dicts):
        with h5py.File(h5, 'r') as f:
            if h5_path not in f:
                continue
            shape = tuple(f[h5_path].shape[1:])
        uuid = (meta or {}).get('uuid', 'no uuid in its .h5')
        shapes.append((shape, uuid, h5))
    return shapes


def _frame_size(shape):
    return ' x '.join(str(n) for n in shape)


def _frame_size_rows(shapes):
    width = max(len(_frame_size(shape)) for shape, _, _ in shapes)
    return '\n'.join(f'  {_frame_size(shape):<{width}}  {uuid}  {h5}' for shape, uuid, h5 in sorted(shapes))


#: Where an extraction's crop_size comes from, as the frame-size messages say it.
CROP_SIZE_SOURCES = (
    'crop_size in config.yaml, or --crop-size on the command line; a recording whose settings '
    'were saved in the arena tool takes crop_size from its entry in session_config.yaml'
)


class FrameCountError(StepInputError, ValueError):
    """A session's extraction holds a different number of frames and timestamps."""


def check_frame_counts(h5s, h5_path='/frames'):
    """
    Stop before PCA when a session's extraction holds a different number of
    frames and timestamps. PCA places each dropped frame from the timestamps,
    one per frame, so without that match the dropped frames cannot be placed
    and the session's timing is unknown. An extraction without timestamps is
    not checked here (PCA times its frames from its frame rate).

    Args:
    h5s (list): paths to the sessions' extracted h5 files.
    h5_path (str): path to the frames inside each h5 file.
    """
    for h5 in h5s:
        with h5py.File(h5, 'r') as f:
            path = timestamps_path(f)
            if path is None or h5_path not in f:
                continue
            n_frames, n_timestamps = len(f[h5_path]), len(f[path])
        if n_frames != n_timestamps:
            raise FrameCountError(
                f'The extraction {h5} holds {n_frames} frames ({h5_path}) and {n_timestamps} timestamps '
                f'({path}). Each frame needs one timestamp, so its dropped frames cannot be placed and its '
                f'timing is unknown. Extract that session again, then run this step again.'
            )


def check_frame_shapes(h5s, dicts, h5_path='/frames'):
    """
    Stop before PCA when the sessions' frames are not all the same size. Every
    session's frames go into one matrix whose columns are a frame's pixels, so
    frames of different sizes (extractions made with different crop_size
    settings) cannot share one set of components.

    Args:
    h5s (list): paths to the sessions' extracted h5 files.
    dicts (list): the sessions' yaml contents, in the same order, for their UUIDs.
    h5_path (str): path to the frames inside each h5 file.
    """
    shapes = _frame_shapes(h5s, dicts, h5_path)

    if len({shape for shape, _, _ in shapes}) <= 1:
        return

    raise ValueError(
        f'The extracted sessions do not all have the same frame size, so they cannot go into '
        f'one PCA. Each session\'s frame size in pixels ({h5_path}), UUID and file:\n'
        f'{_frame_size_rows(shapes)}\n'
        f'Frames of different sizes come from extracting with different crop_size settings. '
        f'Extract every session again with one crop_size ({CROP_SIZE_SOURCES}), then run PCA '
        f'again.'
    )


def check_frames_match_pca(h5s, dicts, components, pca_file, h5_path='/frames'):
    """
    Stop before applying a PCA when sessions' frames are not the size its
    components were trained on. Each component holds one value per pixel of a
    frame, so a PCA can only score frames with that many pixels.

    Args:
    h5s (list): paths to the sessions' extracted h5 files.
    dicts (list): the sessions' yaml contents, in the same order, for their UUIDs.
    components (numpy.ndarray): the trained components, one per row.
    pca_file (str): the PCA file the components were read from, for the message.
    h5_path (str): path to the frames inside each h5 file.
    """
    pixels = int(np.shape(components)[-1])
    wrong = [(shape, uuid, h5) for shape, uuid, h5 in _frame_shapes(h5s, dicts, h5_path)
             if int(np.prod(shape)) != pixels]
    if not wrong:
        return

    side = int(round(np.sqrt(pixels)))
    trained = f'{pixels} pixels' + (f' (frames of {side} x {side})' if side * side == pixels else '')
    raise ValueError(
        f'The PCA in {pca_file} was trained on frames of {trained}, and '
        f'{len(wrong)} session{"s" if len(wrong) != 1 else ""} '
        f'{"have" if len(wrong) != 1 else "has"} frames of another size, so '
        f'{"they" if len(wrong) != 1 else "it"} cannot be scored with it. Each such session\'s '
        f'frame size in pixels ({h5_path}), UUID and file:\n{_frame_size_rows(wrong)}\n'
        f'Frames of different sizes come from extracting with different crop_size settings. '
        f'Apply a PCA trained on extractions made with the same crop_size as these sessions, or '
        f'extract these sessions again with the crop_size the PCA was trained on '
        f'({CROP_SIZE_SOURCES}).'
    )


def check_timestamps(h5s):
    """
    Helper function to determine whether timestamps and/or metadata is missing from
    extracted files. Function will emit a warning if either pieces of data are missing.

    Args:
    h5s (list): List of paths to all extracted h5 files.
    """

    for h5 in h5s:
        try:
            with h5py.File(h5, 'r') as f:
                h5_timestamp_path = timestamps_path(f)
        except Exception:
            h5_timestamp_path = None
        if h5_timestamp_path is None:
            warnings.warn(f'Autoload timestamps for session {h5} failed.')
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


def set_dask_config(memory={'target': 0.85, 'spill': False, 'pause': False, 'terminate': 0.95}):
    """
    Set initial dask configuration parameters

    Args:
    memory (dict): dictionary containing default dask configuration variables to ensure safe amount of resource usage.
    """

    memory = {f'distributed.worker.memory.{k}': v for k, v in memory.items()}
    dask.config.set(memory)
    dask.config.set({'optimization.fuse.ave-width': 5})


#: Where the kernel says which cgroup this process is in, and where the cgroup
#: filesystem is mounted. Module attributes so tests can point them at fakes.
PROC_SELF_CGROUP = '/proc/self/cgroup'
CGROUP_ROOT = '/sys/fs/cgroup'

#: A cgroup memory limit at or above this is cgroup v1's way of saying "no
#: limit" (it reports 9223372036854771712).
CGROUP_NO_LIMIT = 2 ** 60

#: Bytes in each unit SLURM gives memory in; SLURM's units are binary.
SLURM_MEMORY_UNITS = {'K': 2 ** 10, 'M': 2 ** 20, 'G': 2 ** 30, 'T': 2 ** 40}


def _read_number(path, key=None):
    """
    The number in a cgroup file, or the value of key in a file of "key value"
    lines. None when the file is missing or holds no number (cgroup v2's "max").
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    if key is not None:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == key:
                text = parts[1]
                break
        else:
            return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def cgroup_memory_available():
    """
    The memory, in bytes, this process's cgroup still lets it use, or None when
    no cgroup limits it. This is how a container's memory limit (Docker's
    --memory, Kubernetes' limits) and a SLURM job's limit reach a process.

    The tightest limit along the path from the process's cgroup to the root
    binds. What is available under it is the limit less what the group already
    uses, not counting inactive file cache, which the kernel reclaims before it
    runs out (the same "working set" docker stats reports). Both cgroup v2
    (memory.max) and v1 (memory.limit_in_bytes) are read.

    Returns:
    available (int or None): bytes available under the cgroup's limit.
    """
    try:
        with open(PROC_SELF_CGROUP) as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    folders = []
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == '':
            folders.append((CGROUP_ROOT, path, 'memory.max', 'memory.current', 'inactive_file'))
        elif 'memory' in controllers.split(','):
            folders.append((join(CGROUP_ROOT, 'memory'), path, 'memory.limit_in_bytes',
                            'memory.usage_in_bytes', 'total_inactive_file'))

    available = []
    for root, path, limit_file, usage_file, inactive_key in folders:
        # walk from the process's group up to the root of the hierarchy; inside a
        # container with its own cgroup namespace the path is "/" and only the
        # root is there
        parts = [p for p in path.split('/') if p]
        for depth in range(len(parts), -1, -1):
            folder = join(root, *parts[:depth])
            limit = _read_number(join(folder, limit_file))
            if limit is None or limit >= CGROUP_NO_LIMIT:
                continue
            usage = _read_number(join(folder, usage_file)) or 0
            inactive = _read_number(join(folder, 'memory.stat'), inactive_key) or 0
            available.append(max(0, limit - max(0, usage - inactive)))

    return min(available) if available else None


def _parse_slurm_memory(value, cpus):
    """
    Bytes in a memory value as sacct's ReqMem gives it: a number with a unit
    (K, M, G or T; M when there is none), and in older SLURM versions a
    trailing "c" for per-CPU or "n" for per-node.
    """
    value = value.strip()
    per_cpu = value.endswith('c')
    if value[-1:] in ('c', 'n'):
        value = value[:-1]
    unit = value[-1:].upper()
    if unit in SLURM_MEMORY_UNITS:
        number = float(value[:-1]) * SLURM_MEMORY_UNITS[unit]
    else:
        number = float(value) * SLURM_MEMORY_UNITS['M']
    return number * cpus if per_cpu else number


def _slurm_cpu_and_mem(job_id):
    """
    The CPUs and bytes of memory a SLURM job was given, from the variables
    SLURM sets in the job's environment, or from sacct when they are absent.
    Either may be None when SLURM does not say.
    """
    cpus = os.environ.get('SLURM_CPUS_ON_NODE')
    cpus = int(cpus) if cpus and cpus.isdigit() else None
    per_node = os.environ.get('SLURM_MEM_PER_NODE')
    per_cpu = os.environ.get('SLURM_MEM_PER_CPU')

    if per_node and per_node.isdigit() and int(per_node) > 0:
        return cpus, int(per_node) * SLURM_MEMORY_UNITS['M']
    if per_cpu and per_cpu.isdigit() and int(per_cpu) > 0 and cpus is not None:
        return cpus, int(per_cpu) * cpus * SLURM_MEMORY_UNITS['M']

    click.echo('Detected slurm environment, using "sacct" to detect cpu and memory requirements')
    cmd = f'sacct -j {job_id} --format AllocCPUS,ReqMem -X -n -p'
    output = subprocess.check_output(cmd.split(' '))
    alloc_cpus, reqmem = output.decode('utf-8').strip().split('|')[:2]
    alloc_cpus = int(alloc_cpus)
    return alloc_cpus, _parse_slurm_memory(reqmem, alloc_cpus)


def get_env_cpu_and_mem():
    """
    The memory and CPUs this process may use, to size the local dask cluster from.

    The machine's own numbers (its available memory and CPU count) describe the
    whole computer, which a process in a container, on a shared server or in a
    SLURM job does not have to itself. Each is capped by what the process may
    use: memory by its cgroup's limit and, in a SLURM job, by the job's
    allocation; CPUs by the ones it may run on (os.sched_getaffinity) and, in a
    SLURM job, by SLURM_CPUS_ON_NODE.

    Returns:
    mem (float): bytes of memory to give the dask cluster: 80% of what is
        available, leaving room for this process and others it starts
    cpu (int): CPUs to give dask: one fewer than are available, and at least one
    """

    mem = psutil.virtual_memory().available
    cpus = psutil.cpu_count()
    if hasattr(os, 'sched_getaffinity'):
        cpus = min(cpus, len(os.sched_getaffinity(0)))

    cgroup_mem = cgroup_memory_available()
    if cgroup_mem is not None:
        mem = min(mem, cgroup_mem)

    job_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')
    if job_id:
        slurm_cpus, slurm_mem = _slurm_cpu_and_mem(job_id)
        if slurm_mem is not None:
            mem = min(mem, slurm_mem)
        if slurm_cpus is not None:
            cpus = min(cpus, slurm_cpus)

    # account for additional processes that need memory
    return mem * 0.8, max(1, cpus - 1)


#: The loggers dask, distributed and the web servers under its dashboard write to.
DASK_LOGGERS = ('distributed', 'dask', 'tornado', 'bokeh')
DASK_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class _Forward(logging.Handler):
    """
    Pass records at or above level to another handler. Not a StreamHandler, so
    a LocalCluster starting up (which sets every StreamHandler on the
    distributed logger to WARNING while it runs) leaves its level alone.
    """

    def __init__(self, target, level=logging.NOTSET):
        super().__init__(level)
        self.target = target
        target.setFormatter(logging.Formatter(DASK_LOG_FORMAT))

    def emit(self, record):
        self.target.handle(record)

    def close(self):
        self.target.close()
        super().close()


def _not_cancelled(record):
    """
    False for an error whose exception is a cancelled task. distributed logs one
    at ERROR whenever the progress bar stops following a computation (the
    scheduler's feed to it is cancelled); nothing has failed, so it stays out
    of the console.
    """
    return not (record.exc_info and record.exc_info[0] is not None
                and issubclass(record.exc_info[0], asyncio.CancelledError))


@contextmanager
def dask_messages_to(log_file):
    """
    Send dask's log messages and warnings from WARNING up to ERROR to log_file
    instead of the console, for as long as the block runs. Errors go to both,
    so a real failure still shows; the console is told once where the rest went.

    distributed logs freely to the console (worker memory warnings, a dashboard
    port already in use, shutdown chatter), and next to a command's own output
    users could not tell those from a failure. Only this process's loggers are
    moved: workers started as separate processes (local_processes True, or a
    SLURM cluster) log where they always have.

    Args:
    log_file (str): path to the command's log file; it is appended to.
    """
    # warnings and above: the INFO chatter of workers starting and stopping is
    # what a LocalCluster hides from the console too
    file_handler = _Forward(logging.FileHandler(log_file), logging.WARNING)
    console = _Forward(logging.StreamHandler(sys.stderr), logging.ERROR)
    console.addFilter(_not_cancelled)

    def prints(handler):
        return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)

    # the top-level loggers, and every logger under them that prints itself
    # (distributed gives several their own console handler and stops them
    # propagating); other handlers, such as the in-memory one a worker keeps
    # its recent logs in, stay where they are
    names = {name for name, logger in logging.Logger.manager.loggerDict.items()
             if isinstance(logger, logging.Logger) and name.split('.')[0] in DASK_LOGGERS
             and any(prints(h) for h in logger.handlers)}
    saved = {}
    for name in sorted(names | set(DASK_LOGGERS)):
        logger = logging.getLogger(name)
        saved[name] = (list(logger.handlers), logger.propagate)
        for handler in [h for h in logger.handlers if prints(h)]:
            logger.removeHandler(handler)
        logger.addHandler(file_handler)
        logger.addHandler(console)
        logger.propagate = False

    show = warnings.showwarning
    packages = tuple(os.sep + name + os.sep for name in DASK_LOGGERS)

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if any(p in filename for p in packages):
            file_handler.handle(logging.LogRecord(
                'py.warnings', logging.WARNING, filename, lineno,
                f'{category.__name__}: {message}', None, None))
        else:
            show(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    click.echo(f'Dask\'s messages are written to {log_file}; errors are shown here too.')
    try:
        yield
    finally:
        warnings.showwarning = show
        for name, (handlers, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.handlers = handlers
            logger.propagate = propagate
        file_handler.close()


def local_workers(nworkers):
    """
    How many workers a local dask cluster starts and the memory each is given,
    from the memory and CPUs this process may use (get_env_cpu_and_mem).

    Args:
    nworkers (int): the number of workers asked for.

    Returns:
    nworkers (int): the number of workers to start, at most the one asked for.
    mem_limit (float): bytes of memory each worker may use.
    """
    max_mem, max_cpu = get_env_cpu_and_mem()
    overhead = 0.8e9  # memory overhead for each worker; approximate

    # allocating 0.4 of the maximum memory to account for overhead per worker
    allowed = max_mem * 0.4
    max_workers = allowed // overhead

    # Set the number of workers to the maximum the memory and the CPUs
    # allow, and never to none. A cluster with no workers accepts the graph
    # and then never finishes it, so asking for zero is an indefinite hang
    # with no error, and the ceilings reach zero on an ordinary machine:
    # max_workers is zero whenever less than about 2.5GB is free.
    nworkers = int(max(1, min(nworkers, max_workers, max_cpu)))
    mem_limit = max(1, max_mem / nworkers)
    return nworkers, mem_limit


#: Training data above this many times the workers' memory warns: dask spills
#: what does not fit to disk, and training is slower, by how much depending on
#: that disk. Measured on the test suite's 20-session bundle (719,372 frames,
#: 18.4 GB as float32) with the dask cache on a local disk, spilling 9 times
#: over finished in about 1.7 times as long. However far over, the run stops
#: only when the disk lacks room for what would be spilled.
TRAINING_MEMORY_WARN = 1.0

#: The suggested train_on_subset and SLURM workers aim at this fraction of the
#: workers' memory, leaving the rest for each worker's own overhead.
TRAINING_MEMORY_TARGET = 0.9


class TrainingMemoryWarning(Warning):
    """
    The PCA training data is larger than the dask workers' memory, so dask
    spills to disk. A Warning rather than a UserWarning, because the notebook's
    PCA cell hides UserWarnings.
    """


class TrainingMemoryError(RuntimeError):
    """
    The PCA training data is larger than the dask workers' memory, and the disk
    under dask_cache_path cannot hold what dask would spill.
    """


def _format_bytes(n):
    """Bytes in decimal units with two decimals, as dask's memory settings write them."""
    for unit, size in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if n >= size:
            return f'{n / size:.2f} {unit}'
    return f'{n:.0f} B'


def _round_down(x, digits=2):
    """x rounded down to two significant figures, so a suggested value stays under its target."""
    if x <= 0:
        return 0.0
    scale = 10 ** (digits - 1 - int(np.floor(np.log10(x))))
    return float(np.floor(x * scale) / scale)


def _free_disk(path):
    """Free bytes on the disk that holds path, or its nearest existing parent."""
    path = abspath(expanduser(str(path)))
    while not exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def check_training_memory(h5s, config_data):
    """
    Before train-pca starts, compare the memory its training data needs with
    what the dask workers have; warn when it is more, saying how to make it
    fit, and stop when the disk cannot hold what dask would spill.

    Training holds every training frame in the workers' memory at once, as
    float32 after cleaning: the mean over all frames must finish before any
    frame can be centred for the SVD. With missing-data PCA the frame masks are
    held too. When that is more than the workers have, dask spills the excess
    to disk at dask_cache_path, and every pass of the SVD reads it back, so
    training slows with how much is spilled and how fast that disk is. When the
    disk lacks room for what would be spilled, the run would fail partway, so
    it stops before starting.

    Args:
    h5s (list): paths to the sessions' extracted h5 files.
    config_data (dict): train-pca's parameters.
    """
    cluster_type = config_data.get('cluster_type', 'local')
    if cluster_type not in ('local', 'slurm'):
        return

    h5_path = config_data.get('h5_path', '/frames')
    subset = config_data.get('train_on_subset', 1)
    lengths, pixels = [], None
    for h5 in h5s:
        with h5py.File(h5, 'r') as f:
            if h5_path not in f:
                continue
            frames = f[h5_path]
            lengths.append(len(frames))
            pixels = tuple(frames.shape[1:])
    if not lengths:
        return

    missing_data = config_data.get('missing_data')
    bytes_per_frame = float(np.prod(pixels)) * (4 + (1 if missing_data else 0))

    def used_frames(fraction):
        return sum(int(n * fraction) for n in lengths)

    n_frames = used_frames(subset)
    needed = n_frames * bytes_per_frame

    nworkers = int(config_data.get('nworkers', 1))
    if cluster_type == 'local':
        n, per_worker = local_workers(nworkers)
        have = n * per_worker
        workers = (f'{n} local worker{"s" if n != 1 else ""} of {_format_bytes(per_worker)}, '
                   f'80% of the memory this process may use')
    else:
        processes = max(1, int(config_data.get('processes', 1)))
        jobs = int(np.ceil(nworkers / processes))
        have = jobs * dask.utils.parse_bytes(config_data['memory'])
        workers = (f'{jobs} SLURM job{"s" if jobs != 1 else ""} of {config_data["memory"]}, '
                   f'from nworkers {nworkers} and memory {config_data["memory"]}')

    ratio = needed / have
    if ratio <= TRAINING_MEMORY_WARN:
        return

    cache = config_data.get('dask_cache_path') or os.getcwd()
    spill = needed - have
    free = _free_disk(cache)
    no_room = free is not None and free < spill

    # a train_on_subset that brings the training data under the workers' memory
    fit_subset = _round_down(subset * TRAINING_MEMORY_TARGET * have / needed)
    fit_frames = used_frames(fit_subset)
    # SLURM workers of the memory already set (train-pca's default when there is none)
    memory = config_data.get('memory') or '15GB'
    per_job = dask.utils.parse_bytes(memory)
    processes = max(1, int(config_data.get('processes', 1)))
    jobs_needed = int(np.ceil(needed / (TRAINING_MEMORY_TARGET * per_job)))
    slurm_workers = jobs_needed * processes

    size = ' x '.join(str(p) for p in pixels)
    what = (f'Training PCA on {n_frames:,} frames of {size} pixels from {len(lengths)} '
            f'session{"s" if len(lengths) != 1 else ""} (train_on_subset {subset:g}, of '
            f'{sum(lengths):,} frames in all) needs about {_format_bytes(needed)} of memory'
            f'{" with the missing-data masks" if missing_data else ""}, and the dask workers have '
            f'{_format_bytes(have)} ({workers}): {ratio:.1f} times what they have.')
    remedies = (
        f'  - Train on fewer frames: set train_on_subset to {fit_subset:g} '
        f'({fit_frames:,} frames, about {_format_bytes(fit_frames * bytes_per_frame)}).\n'
        f'  - If you have a SLURM cluster, run the workers there: set cluster_type slurm, '
        f'nworkers {slurm_workers}, memory {memory} per worker '
        f'({_format_bytes(jobs_needed * per_job)} in all), and queue to your cluster\'s partition.\n'
        f'In the notebook\'s PCA cell these are config_data[\'train_on_subset\'], '
        f'config_data[\'cluster_type\'], config_data[\'nworkers\'], config_data[\'memory\'] and '
        f'config_data[\'queue\']; on the command line, --train-on-subset, --cluster-type, '
        f'--nworkers, --memory and --queue.'
    )

    if no_room:
        raise TrainingMemoryError(
            f'{what} Dask would spill about {_format_bytes(spill)} to disk in {cache}, which has '
            f'{_format_bytes(free)} free, so the run would fail partway. It has been stopped before '
            f'starting. Set dask_cache_path to a folder on a disk with more than '
            f'{_format_bytes(spill)} free, or make the training data fit in memory with one of '
            f'these:\n{remedies}'
        )
    warnings.warn(
        f'{what} Dask will spill about {_format_bytes(spill)} to disk in {cache} and read it back '
        f'on every pass of the SVD, so training is slower, by how much depending on that disk: '
        f'measured on a fast local disk at about twice as long or less, and on a network or '
        f'slow disk it can take many hours. To keep it in memory, do one of these:\n{remedies}',
        TrainingMemoryWarning, stacklevel=2,
    )


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

        requested = nworkers
        nworkers, mem_limit = local_workers(nworkers)
        if nworkers < requested:
            click.echo(f'Reducing number of workers to {nworkers} to account for worker base memory and the number of CPUs')

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
                     baseline=True, timestamps=None, session=None):
    """
    Compute changepoints and its corresponding distribution. Changepoints describe
    the magnitude of frame-to-frame changes of mouse pose.

    Args:
    scores (numpy.ndarray): nframes * rows * columns
    k (int): klags - Lag to use for derivative calculation.
    sigma (float): Half-width of the smoothing window in frames, rounded down to whole frames. The Gaussian inside it
    has a standard deviation of 1.5 frames. Also sets how many frames are trimmed from each end of the changepoint
    curve (6 * sigma).
    peak_height (float): user-defined peak Changepoint length.
    peak_neighbors (int): number of peaks in the CP curve.
    baseline (bool): normalize data.
    timestamps (numpy.array): loaded timestamps.
    session (str): the session's uuid and file, named if its timestamps give no frame rate.

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
    # frames outside the lag window have no squared difference
    normed_df[:, :k // 2] = np.nan
    normed_df[:, -k // 2:] = np.nan
    if sigma is not None and sigma > 0:
        edge = int(6 * sigma)
        if edge > 0:
            normed_df[:, :edge] = np.nan
            normed_df[:, -edge:] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normed_df = np.nanmean(normed_df, axis=0)

        if baseline:
            normed_df -= np.nanmin(normed_df)

        if timestamps is not None:
            frame_rate_from_timestamps(timestamps, session)
            normed_df, _, _ = insert_nans(timestamps, normed_df)

        normed_df = np.squeeze(normed_df)
        cps = scipy.signal.argrelextrema(
            normed_df, np.greater, order=peak_neighbors)[0]
        cps = cps[np.argwhere(normed_df[cps] > peak_height)]

    return cps, normed_df

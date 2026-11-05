"""
General utility functions throughout the extract package.
"""
import os
import re
import cv2
import math
import json
import h5py
import click
import tarfile
import warnings
import numpy as np
from copy import deepcopy
from moseq2_common.frames import odd_kernel
from moseq2_extract.io.image import write_image
from moseq2_extract.io.video import get_movie_info
from os.path import join, exists, splitext, basename, abspath, dirname

#: Where graduate_walls centres its wall gradient, in pixels from the bucket's centre (x, y).
X_BG_OFFSET = -2
Y_BG_OFFSET = 2
#: Whether graduate_walls keeps the measured floor inside the bucket rather than the gradient's.
FLOOR_SLANT = False


def filter_warnings(func):
    """
    Applies warnings.simplefilter() to ignore warnings when
     running the main gui functionaity in a Jupyter Notebook.
     The function will filter out: yaml.error.UnsafeLoaderWarning, FutureWarning and UserWarning.

    Args:
    func (function): function to silence enclosed warnings.

    Returns:
    apply_warning_filters (func): Returns passed function after warnings filtering is completed.
    """
    def apply_warning_filters(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)
            return func(*args, **kwargs)
    return apply_warning_filters


def set_bground_to_plane_fit(bground_im, plane, output_dir):
    """
    Replaces median-computed background image with plane fit.
    Only occurs if config_data['use_plane_bground'] == True.

    The plane is written to ``plane_bground.tiff``. ``bground.tiff`` in the same folder is the
    cached median background that the arena is found on, so the plane never replaces it.

    Args:
    bground_im (numpy.ndarray): Background image computed via median value in each pixel of depth video.
    plane (numpy.ndarray): Computed ROI Plane using RANSAC.
    output_dir (str): Path to write the plane-fit background image to.

    Returns:
    bground_im (numpy.ndarray): The background image.
    """

    xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
    coords = np.vstack((xx.ravel(), yy.ravel()))

    plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
    plane_im = plane_im.reshape(bground_im.shape)

    write_image(join(output_dir, 'plane_bground.tiff'), plane_im, scale=True)

    return plane_im

def get_frame_range_indices(trim_beginning, trim_ending, nframes):
    """
    Compute the total number of frames to be extracted, and find the start and end indices.

    Args:
    trim_beginning (int): number of frames to remove from beginning of recording
    trim_ending (int): number of frames to remove from ending of recording
    nframes (int): total number of requested frames to extract

    Returns:
    nframes (int): total number of frames to extract
    first_frame_idx (int): index of the frame to begin extraction from
    last_frame_idx (int): index of the last frame in the extraction
    """
    assert all((trim_ending >= 0, trim_beginning >= 0)) , "frame_trim arguments must be greater than or equal to 0!"

    first_frame_idx = 0
    if trim_beginning > 0 and trim_beginning < nframes:
        first_frame_idx = trim_beginning

    last_frame_idx = nframes
    if first_frame_idx < (nframes - trim_ending) and trim_ending > 0:
        last_frame_idx = nframes - trim_ending

    total_frames = last_frame_idx - first_frame_idx

    return total_frames, first_frame_idx, last_frame_idx

def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    """
    Generates batches used to chunk videos prior to extraction.

    Args:
    nframes (int): total number of frames
    chunk_size (int): the number of desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns:
    out (list): the list of batches
    """

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out

def _timestamp_field(field):
    """
    Read one field of a timestamp file as a float. The columns are separated by
    whitespace, so a single comma in a field without a decimal point is a decimal
    separator, as a system set to a decimal-comma locale writes it.
    """
    if isinstance(field, bytes):
        field = field.decode()
    if field.count(',') == 1 and '.' not in field:
        field = field.replace(',', '.')
    return float(field)

def load_timestamps(timestamp_file, col=0, alternate=False):
    """
    Read timestamps from space delimited text file for timestamps. A decimal comma
    (``25044816,2766``) is read as a decimal point.

    Args:
    timestamp_file (str): path to timestamp file
    col (int): column in ts file read.
    alternate (boolean): specified if timestamps were saved in a csv file. False means txt file and True means csv file.

    Returns:
    ts (1D array): list of timestamps
    """

    def read(lines, name):
        ts = []
        for line_number, line in enumerate(lines, start=1):
            cols = line.split()
            try:
                ts.append(_timestamp_field(cols[col]))
            except ValueError:
                raise ValueError(
                    f'Could not read a timestamp from line {line_number} of {name}: '
                    f'column {col} holds {cols[col]!r}, which is not a number. Each line should '
                    f'hold whitespace-separated numbers, such as "674717230.8241 0". '
                    f'Check that the file is the recording\'s own timestamp file and is not damaged.'
                ) from None
        return np.array(ts)

    try:
        with open(timestamp_file, 'r') as f:
            ts = read(f, timestamp_file)
    except TypeError as e:
        # try iterating directly
        ts = read(timestamp_file, getattr(timestamp_file, 'name', 'the timestamp file'))
    except FileNotFoundError as e:
        ts = None
        warnings.warn('Timestamp file was not found! Make sure the timestamp file exists is named '
            '"depth_ts.txt" or "timestamps.csv".')
        warnings.warn('This could cause issues for large number of dropped frames during the PCA step while '
            'imputing missing data.')

    # if timestamps were saved in a csv file
    if alternate:
        ts = ts * 1000

    return ts

def detect_avi_file(finfo):
    """
    Detect the camera type by comparing the read video resolution with known
     outputted dimensions of different camera types.

    Args:
    finfo (dict): dictionary containing the file metadata,

    Returns:
    detected (str): name of the detected camera type.
    """

    detected = 'azure'
    potential_camera_dims = {
        'kinect': [[512, 424]],
        'realsense': [[640, 480]],
        'azure': [[640, 576],
                  [320, 288],
                  [512, 512],
                  [1024, 1024]]
    }

    # Check dimensions
    if finfo is not None:
        if list(finfo['dims']) in potential_camera_dims['azure']:
            # Default Azure dimensions
            detected = 'azure'
        elif list(finfo['dims']) in potential_camera_dims['realsense']:
            # Realsense D415 output dimensions
            detected = 'realsense'
        elif list(finfo['dims']) in potential_camera_dims['kinect']:
            # Kinect output dimensions
            detected = 'kinect'
        else:
            warnings.warn('Could not infer camera type, using default Azure parameters.')

    return detected

def detect_and_set_camera_parameters(config_data, input_file=None):
    """
    Set bg_roi_weights, pixel_format and movie_dtype to the values for the camera named by
    camera_type, or, when it is ``auto``, for the camera told from the file. A camera_type
    that names no known camera (None included) changes nothing and warns.

    Only a setting still at the generic default (the Kinect v2's value, which a generated
    config holds) or missing is replaced; a value the user changed is kept. ``k2`` is the
    Kinect v2, as the notebook and moseq2-pca name it.

    Only Azure Kinect and Orbbec Femto cameras write ``.mkv``, so an ``.mkv`` recording
    gets the Azure settings whatever ``camera_type`` says. Notebooks from 1.4.5 and
    earlier left ``k2`` or ``auto`` in an Azure project's config; where it says ``k2`` or
    ``kinect``, a line names the recording and says to set ``camera_type: azure``.

    Args:
    config_data (dict): dictionary containing all input parameters.
    input_file (str): path to raw depth file

    Returns:
    config_data (dict): updated dictionary with bg-roi-weights to use for extraction.
    """

    # Auto-setting background weights
    stated = config_data.get('camera_type')
    camera_type = 'kinect' if stated == 'k2' else stated
    finfo = config_data.get('finfo')
    is_mkv = isinstance(input_file, (str, os.PathLike)) and str(input_file).endswith('.mkv')

    default_parameters = {
        'kinect': {
            'bg_roi_weights': (1, .1, 1),
            'pixel_format': 'gray16le',
            'movie_dtype': '<u2'
        },
        'azure': {
            'bg_roi_weights': (10, 0.1, 1),
            'pixel_format': 'gray16be',
            'movie_dtype': '>u2'
        },
        'realsense': {
            'bg_roi_weights': (10, 0.1, 1),
            'pixel_format': 'gray16le',
            'movie_dtype': '<u2',
        },
    }

    if type(input_file) is tarfile.TarFile:
        detected = 'kinect'
    elif camera_type == 'auto' and input_file is not None:
        if input_file.endswith('.dat'):
            detected = 'kinect'
        elif input_file.endswith('.mkv'):
            detected = 'azure'
        elif input_file.endswith('.avi'):
            if finfo is None:
                finfo = get_movie_info(input_file,
                                       mapping=config_data.get('mapping', 0),
                                       threads=config_data.get('threads', 8)
                                       )
            detected = detect_avi_file(finfo)
        else:
            warnings.warn('Extension not recognized, trying default Kinect v2 parameters.')
            detected = 'kinect'

        _fill_camera_defaults(config_data, default_parameters[detected], default_parameters['kinect'])
    elif camera_type == 'kinect' and is_mkv:
        _fill_camera_defaults(config_data, default_parameters['azure'], default_parameters['kinect'])
        used = ', '.join(f'{key} {_setting_text(config_data[key])}' for key in default_parameters['azure'])
        print(f'{input_file} is an .mkv recording, which only Azure Kinect and Orbbec Femto cameras '
              f'write, but config.yaml says camera_type: {stated} (the Kinect v2). Used the Azure '
              f'settings for it: {used}. Set camera_type: azure in config.yaml.')
    elif camera_type in default_parameters:
        _fill_camera_defaults(config_data, default_parameters[camera_type], default_parameters['kinect'])
    else:
        warnings.warn('Warning, make sure the following parameters are set to best handle your camera type: '
                      '"bg_roi_weights", "pixel_format", "movie_dtype"')

    return config_data


def _fill_camera_defaults(config_data, camera_defaults, generic_defaults):
    """Set each camera default whose setting is missing or still at the generic default."""
    for key, value in camera_defaults.items():
        current = config_data.get(key)
        if current is None or _same_setting(current, generic_defaults[key]):
            config_data[key] = value


def _setting_text(value):
    """A setting as a user would write it in config.yaml: ``[10, 0.1, 1]``, ``gray16be``."""
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(str(v) for v in value)}]"
    return str(value)


def _same_setting(a, b):
    """Compare two settings, treating a list read back from yaml and a tuple as equal."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b

def check_filter_sizes(config_data):
    """
    Round each even spatial and temporal median kernel size up to the next odd one.

    Args:
    config_data (dict): a dictionary holding all extraction parameters

    Returns:
    config_data (dict): a copy of ``config_data`` whose ``spatial_filter_size`` and
        ``temporal_filter_size`` are new lists, each size through ``moseq2_common.frames.odd_kernel``
    """
    return {
        **config_data,
        "spatial_filter_size": [odd_kernel(size) for size in config_data["spatial_filter_size"]],
        "temporal_filter_size": [odd_kernel(size) for size in config_data["temporal_filter_size"]],
    }

def load_metadata(metadata_file):
    """
    Load a session's metadata.json, or its stand-in when the recording has none.

    A recording without a metadata.json gets an empty ``SubjectName``, its folder's name as
    ``SessionName`` and an empty ``StartTime``, in memory only: nothing is written into the
    recording's folder. A corrected metadata.json takes effect when the recording is extracted again.

    Args:
    metadata_file (str or file object): path to metadata.json, or the file opened (inside an archive).

    Returns:
    metadata (dict): metadata dictionary of JSON contents
    """

    if not isinstance(metadata_file, (str, os.PathLike)):
        return json.load(metadata_file)
    if not exists(metadata_file):
        return {'SubjectName': '', 'SessionName': basename(dirname(abspath(metadata_file))), 'StartTime': ''}
    with open(metadata_file, 'r') as f:
        return json.load(f)


def scalar_attributes():
    """
    Gets scalar attributes dict with names paired with descriptions.

    Returns:
    attributes (dict): a dictionary of metadata keys and descriptions.
    """

    attributes = {
        'centroid_x_px': 'X centroid (pixels)',
        'centroid_y_px': 'Y centroid (pixels)',
        'velocity_2d_px': '2D velocity (pixels / frame), note that missing frames are not accounted for',
        'velocity_3d_px': '3D velocity (pixels / frame), note that missing frames are not accounted for, also height is in mm, not pixels for calculation',
        'width_px': 'Mouse width (pixels)',
        'length_px': 'Mouse length (pixels)',
        'area_px': 'Mouse area (pixels)',
        'centroid_x_mm': 'X centroid (mm)',
        'centroid_y_mm': 'Y centroid (mm)',
        'velocity_2d_mm': '2D velocity (mm / frame), note that missing frames are not accounted for',
        'velocity_3d_mm': '3D velocity (mm / frame), note that missing frames are not accounted for',
        'width_mm': 'Mouse width (mm)',
        'length_mm': 'Mouse length (mm)',
        'area_mm': 'Mouse area (mm^2)',
        'height_ave_mm': 'Mouse average height (mm)',
        'angle': 'Angle (radians, unwrapped)',
        'velocity_theta': 'Angular component of velocity (arctan(vel_x, vel_y))'
    }

    return attributes


def convert_raw_to_avi_function(input_file, chunk_size=2000, fps=30, delete=False, threads=3):
    """
    Compress depth file (.dat, '.mkv') to avi file.

    Args:
    input_file (str): path to depth file
    chunk_size (int): size of chunks to process at a time
    fps (int): frames per second
    delete (bool): flag for deleting original depth file
    threads (int): number of threads to write video.

    """

    new_file = f'{splitext(input_file)[0]}.avi'
    print(f'Converting {input_file} to {new_file}')
    # turn into os system call...
    use_kwargs = {
        'output-file': new_file,
        'chunk-size': chunk_size,
        'fps': fps,
        'threads': threads
    }
    use_flags = {
        'delete': delete
    }
    base_command = f'moseq2-extract convert-raw-to-avi {input_file}'
    for k, v in use_kwargs.items():
        base_command += f' --{k} {v}'
    for k, v in use_flags.items():
        if v:
            base_command += f' --{k}'

    print(base_command)
    print()

    os.system(base_command)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    Create subarrays of an array with a given stride and window length.

    Args:
    a (np.ndarray) - original array
    L (int) - Window Length
    S (int) - Stride size

    Returns:
    (np.ndarray) - array of subarrays
    """

    # https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052

    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def dict_to_h5(h5, dic, root='/', annotations=None):
    """
    Save an dict to an h5 file, mounting at root.
    Keys are mapped to group names recursively.

    Args:
    h5 (h5py.File instance): h5py.file object to operate on
    dic (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.
    
    """

    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5[dest] = item
            elif isinstance(item, (tuple, list)):
                h5[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5[dest] = np.asarray([item])[0]
            elif item is None:
                h5.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
            elif isinstance(item, dict):
                dict_to_h5(h5, item, dest)
            else:
                raise ValueError('Cannot save {} type to key {}'.format(type(item), dest))
        except Exception as e:
            print(e)
            if key != 'inputs':
                print('h5py could not encode key:', key)

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs['description'] = ""
            else:
                h5[dest].attrs['description'] = annotations[key]


def escape_path(path):
    """
    Return a path to return to original base directory.

    Args:
    path (str): path to current working dir

    Returns:
    path (str): path to original base_dir
    """

    return re.sub(r'\s', r'\ ', path)


def load_textdata(data_file, dtype=np.float32):
    """
    Loads timestamp from txt/csv file.

    Args:
    data_file (str): path to timestamp file
    dtype (dtype): data type of timestamps

    Returns:
    data (np.ndarray): timestamp data
    timestamps (numpy.array): the array for the timestamps
    """

    data = []
    timestamps = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = line.split(' ', 1)
            # appending timestamp value
            timestamps.append(int(float(tmp[0])))

            # append data indicator value
            #
            # Parsed as float and then cast, rather than parsed at `dtype`.
            # numpy 2.4 stopped accepting a decimal string for an integer
            # dtype -- `np.fromstring("0.000000", sep=",", dtype=int)` raises
            # ValueError where 2.2 returned 0 -- and these files hold decimals
            # whatever the column means, so `predictions.txt` stopped loading.
            # extraction catches that and leaves the entry out, so the data went
            # missing with nothing raised.
            clean_data = np.array(
                [piece for piece in tmp[1].strip().split(',') if piece],
                dtype=float).astype(dtype)
            data.append(clean_data)

    data = np.stack(data, axis=0).squeeze()
    timestamps = np.array(timestamps, dtype=int)

    return data, timestamps


def mouse_threshold_filter(h5file, thresh=0):
    """
    Filter frames in h5 files by threshold value.

    Args:
    h5file (str): path to h5 file
    thresh (int): threshold at which to apply filter

    Returns:
    (3d-np boolean array): array of regions to include after threshold filter.
    """

    with h5py.File(h5file, 'r') as f:
        # select 1st 1000 frames
        frames = f['frames'][:min(f['frames'].shape[0], 1000)]
    return np.nanmean(frames) > thresh


def click_param_annot(click_cmd):
    """
    Return a dict that maps option names to help strings from a click.Command instance.

    Args:
    click_cmd (click.Command): command to annotate

    Returns:
    annotations (dict): dictionary of options and their help messages
    """

    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations

def get_bucket_center(img, true_depth, threshold=650):
    """
    Find Centroid coordinates of circular bucket.

    Args:
    img (np.ndaarray): original background image.
    true_depth (float): distance value from camera to bucket floor (automatically pre-computed)
    threshold (float): distance values to accept region into detected circle. (used to reduce fall noise interference)

    Returns:
    cX (int): x-coordinate of circle centroid
    cY (int): y-coordinate of circle centroid
    """

    # https://stackoverflow.com/questions/19768508/python-opencv-finding-circle-sun-coordinates-of-center-the-circle-from-pictu
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img, threshold, true_depth, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

def _floor_depth(bground_im, cx, cy, half_width=2):
    """
    Read the floor depth around the arena centre found by get_bucket_center.

    Args:
    bground_im (np.ndarray): background image
    cx (int): x-coordinate of the arena centre
    cy (int): y-coordinate of the arena centre
    half_width (int): the patch read is (2 * half_width + 1) pixels square

    Returns:
    depth (float): median depth of the pixels in the patch that have a reading (a depth of 0 is "no reading")
    """

    patch = bground_im[max(cy - half_width, 0):cy + half_width + 1, max(cx - half_width, 0):cx + half_width + 1]
    observed = patch[patch > 0]
    if observed.size == 0:
        raise ValueError(
            'Could not read the floor depth at the arena centre: no pixel near it has a depth reading. '
            'Set "bg_roi_depth_range" by hand and "manual_set_depth_range" to True.'
        )
    return float(np.median(observed))

def make_gradient(width, height, h, k, a, b, theta=0):
    """
    Create gradient around bucket floor representing slanted wall values.

    Args:
    width (int): bounding box width
    height (int) bounding box height
    h (int): centroid x coordinate
    k (int): centroid y coordinate
    a (int): x-radius of drawn ellipse
    b (int): y-radius of drawn ellipse
    theta (float): degree to rotate ellipse in radians. (has no effect if drawing a circle)

    Returns:
    np.ndarray: numpy array with weighted values from 0.08 -> 0.8 representing the proportion of values
    to create a gradient from. 0.8 being the proportioned values closest to the circle wall.
    """

    # https://stackoverflow.com/questions/49829783/draw-a-gradual-change-ellipse-in-skimage/49848093#49848093
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a ** 2, b ** 2

    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[-k:height - k, -h:width - h]

    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)

    return np.clip(0.98 - weights, 0, 0.81)


def graduate_dilated_wall_area(bground_im, config_data, strel_dilate, output_dir):
    r"""
    Creates a gradient to represent the dilated (now visible) bucket wall regions.
    Only is used if background is dilated to capture larger rodents in convex shaped buckets (\_/).
    
    Args:
    bground_im (np.ndarray): the computed background image.
    config_data (dict): dictionary containing helper user configuration parameters.
    strel_dilate (cv2.structuringElement): dilation structuring element used to dilate background image;
        None skips the dilation.
    output_dir (str): path to save newly computed background to use.

    Returns:
    bground_im (np.ndarray): the new background image with a gradient around the floor from high to low depth values.
    """

    # store old and new backgrounds
    old_bg = deepcopy(bground_im)

    # dilate background size to match ROI size and attribute wall noise to cancel out
    if strel_dilate is None:
        bground_im = deepcopy(old_bg)
    else:
        bground_im = cv2.dilate(old_bg, strel_dilate, iterations=config_data.get('dilate_iterations', 5))

    # determine center of bground roi
    width, height = bground_im.shape[1], bground_im.shape[0]  # shape of bounding box

    # getting helper user parameters
    true_depth = config_data['true_depth']
    xoffset = X_BG_OFFSET
    yoffset = Y_BG_OFFSET
    widen_radius = config_data.get('widen_radius', 0)
    bg_threshold = np.median(bground_im)

    # getting bground centroid
    cx, cy = get_bucket_center(deepcopy(old_bg), true_depth, threshold=bg_threshold)

    # set up gradient
    h, k = cx + xoffset, cy + yoffset   # centroid of gradient circle
    a, b = cx + widen_radius + 67, cy + widen_radius + 67 # x,y radii of gradient circle
    theta = math.pi/24 # gradient angle; arbitrary - used to rotate ellipses.

    # create slant gradient
    bground_im = np.uint16((make_gradient(width, height, h, k, a, b, theta)) * 255)

    # scale it back to depth
    bground_im = np.uint16((bground_im/bground_im.max())*true_depth)

    # overlay with actual bucket floor distance
    if FLOOR_SLANT:
        ret, thresh = cv2.threshold(old_bg, np.median(old_bg), true_depth, 0)
        contours, _ = cv2.findContours(thresh.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(bground_im.shape, np.uint8)

        cv2.drawContours(mask, contours, -1, (255), -1)

        tmp = np.where(mask == True, bground_im, old_bg)
        bground_im = np.where(tmp == 0, bground_im, tmp)
    else:
        mask = np.ma.equal(old_bg, old_bg.max())
        bground_im = np.where(mask == True, old_bg, bground_im)

    bground_im = cv2.GaussianBlur(bground_im, (7, 7), 7)

    write_image(join(output_dir, 'new_bg.tiff'), bground_im, scale=True)

    return bground_im

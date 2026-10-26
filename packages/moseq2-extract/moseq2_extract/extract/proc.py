"""
Video pre-processing utilities for detecting ROIs and extracting raw data.
"""

import cv2
import tarfile
import scipy.stats
import numpy as np
import scipy.signal
import skimage.measure
import scipy.interpolate
import skimage.morphology
from copy import deepcopy
from tqdm.auto import tqdm
import moseq2_extract.io.video
import moseq2_extract.extract.roi
from os.path import exists, join, dirname
from moseq2_extract.io.image import read_image, write_image
from moseq2_extract.util import convert_pxs_to_mm, strided_app
from moseq2_extract.extract.flip_classifier import (
    load_flip_classifier,
    is_legacy_pickle_failure,
)


def get_flips(frames, flip_file=None, smoothing=None):
    """
    Predict frames where mouse orientation is flipped to later correct.

    Args:
    frames (numpy.ndarray): frames x rows x columns, cropped mouse
    flip_file (str): path to pre-trained scipy random forest classifier
    smoothing (int): kernel size for median filter smoothing of random forest probabilities

    Returns:
    flips (numpy.array):  array for flips
    """

    try:
        clf = load_flip_classifier(flip_file)
    except IOError:
        print(f"Could not open file {flip_file}")
        raise
    except Exception as exc:
        # Only the failure this port introduces is reported differently. A file
        # that is not a classifier failed the same way before the port and has
        # to keep failing the same way, or the port has changed behaviour on an
        # input that never needed changing.
        if not is_legacy_pickle_failure(exc):
            raise
        raise ValueError(
            f"Could not read the flip classifier {flip_file}: "
            f"{type(exc).__name__}: {exc}\n"
            "Classifiers published before 2026 are scikit-learn 0.19 pickles, "
            "which this version cannot read. Download the converted classifier "
            "with: moseq2-extract download-flip-file"
        )

    flip_class = np.where(clf.classes_ == 1)[0]

    probas = clf.predict_proba(
        frames.reshape((-1, frames.shape[1] * frames.shape[2]))
    )

    if smoothing:
        for i in range(probas.shape[1]):
            probas[:, i] = scipy.signal.medfilt(probas[:, i], smoothing)

    flips = probas.argmax(axis=1) == flip_class

    return flips


def get_largest_cc(frames, progress_bar=False):
    """
    Returns largest connected component blob in image

    Args:
    frames (numpy.ndarray): frames x rows x columns, uncropped mouse
    progress_bar (bool): display progress bar

    Returns:
    foreground_obj (numpy.ndarray):  frames x rows x columns, true where blob was found
    """

    foreground_obj = np.zeros((frames.shape), 'bool')

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Computing largest Connected Component'):
        nb_components, output, stats, centroids =\
            cv2.connectedComponentsWithStats(frames[i], connectivity=4)
        szs = stats[:, -1]
        foreground_obj[i] = output == szs[1:].argmax()+1

    return foreground_obj


def get_bground_im_file(frames_file, frame_stride=500, med_scale=5, output_dir=None, **kwargs):
    """
    Load or compute background from file.

    Args:
    frames_file (str): path to the depth video
    frame_stride (int): stride size between frames for median bground calculation
    med_scale (int): kernel size for median blur for background images.
    kwargs (dict): extra keyword arguments

    Returns:
    bground (numpy.ndarray): background image
    """

    if output_dir is None:
        bground_path = join(dirname(frames_file), 'proc', 'bground.tiff')
    else:
        bground_path = join(output_dir, 'bground.tiff')

    if type(frames_file) is not tarfile.TarFile:
        kwargs = deepcopy(kwargs)
    finfo = kwargs.pop('finfo', None)

    # Compute background image if it doesn't exist. Otherwise, load from file
    if not exists(bground_path) or kwargs.get('recompute_bg', False):
        if finfo is None:
            finfo = moseq2_extract.io.video.get_movie_info(frames_file, **kwargs)

        frame_idx = np.arange(0, finfo['nframes'], frame_stride)
        frame_store = []
        for i, frame in enumerate(frame_idx):
            frs = moseq2_extract.io.video.load_movie_data(frames_file,
                                                          [int(frame)], 
                                                          frame_size=finfo['dims'], 
                                                          finfo=finfo, 
                                                          **kwargs).squeeze()
            frame_store.append(cv2.medianBlur(frs, med_scale))

        bground = np.nanmedian(frame_store, axis=0)

        write_image(bground_path, bground, scale=True)
    else:
        bground = read_image(bground_path, scale=True)
        
    return bground


def get_bbox(roi):
    """
    return an array with the x and y boundaries given ROI.

    Args:
    roi (np.ndarray): ROI boolean mask to calculate bounding box.

    Returns:
    bbox (np.ndarray): Bounding Box around ROI
    """

    y, x = np.where(roi > 0)

    if len(y) == 0 or len(x) == 0:
        return None
    else:
        bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])
        return bbox

def threshold_chunk(chunk, min_height, max_height):
    """
    Threshold out depth values that are less than min_height and larger than
    max_height.

    Args:
    chunk (np.ndarray): Chunk of frames to threshold (nframes, width, height)
    min_height (int): Minimum depth values to include after thresholding.
    max_height (int): Maximum depth values to include after thresholding.
    dilate_iterations (int): Number of iterations the ROI was dilated.

    Returns:
    chunk (3D np.ndarray): Updated frame chunk.
    """

    chunk[chunk < min_height] = 0
    chunk[chunk > max_height] = 0

    return chunk

def get_roi(depth_image,
            strel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            dilate_iterations=0,
            erode_iterations=0,
            strel_erode=None,
            noise_tolerance=30,
            bg_roi_weights=(1, .1, 1),
            overlap_roi=None,
            bg_roi_gradient_filter=False,
            bg_roi_gradient_kernel=7,
            bg_roi_gradient_threshold=3000,
            bg_roi_fill_holes=True,
            get_all_data=False,
            **kwargs):
    """
    Compute an ROI using RANSAC plane fitting and simple blob features.

    Args:
    depth_image (np.ndarray): Singular depth image frame.
    strel_dilate (cv2.StructuringElement): dilation shape to use.
    dilate_iterations (int): number of dilation iterations.
    erode_iterations (int): number of erosion iterations.
    strel_erode (int): image erosion kernel size.
    noise_tolerance (int): threshold to use for noise filtering.
    bg_roi_weights (tuple): weights describing threshold to accept ROI.
    overlap_roi (np.ndarray): list of ROI boolean arrays to possibly combine.
    bg_roi_gradient_filter (bool): Boolean for whether to use a gradient filter.
    bg_roi_gradient_kernel (tuple): Kernel size of length 2, e.g. (1, 1.5)
    bg_roi_gradient_threshold (int): Threshold for noise gradient filtering
    bg_roi_fill_holes (bool): Boolean to fill any missing regions within the ROI.
    get_all_data (bool): If True, returns all ROI data, else, only return ROIs and computed Planes
    kwargs (dict) Dictionary containing `bg_roi_depth_range` parameter for plane_ransac()

    Returns:
    rois (list): list of detected roi images.
    roi_plane (np.ndarray): computed ROI Plane using RANSAC.
    bboxes (list): list of computed bounding boxes for each respective ROI.
    label_im (list): list of scikit-image image properties
    ranks (list): list of ROI ranks.
    shape_index (list): list of rank means.
    """

    if bg_roi_gradient_filter:
        gradient_x = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      1, 0, ksize=bg_roi_gradient_kernel))
        gradient_y = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      0, 1, ksize=bg_roi_gradient_kernel))
        mask = np.logical_and(gradient_x < bg_roi_gradient_threshold, gradient_y < bg_roi_gradient_threshold)
    else:
        mask = None

    roi_plane, dists = moseq2_extract.extract.roi.plane_ransac(
        depth_image, noise_tolerance=noise_tolerance, mask=mask, **kwargs)
    dist_ims = dists.reshape(depth_image.shape)

    if bg_roi_gradient_filter:
        dist_ims[~mask] = np.inf

    bin_im = dist_ims < noise_tolerance

    # anything < noise_tolerance from the plane is part of it
    label_im = skimage.measure.label(bin_im)
    region_properties = skimage.measure.regionprops(label_im)

    areas = np.zeros((len(region_properties),))
    extents = np.zeros_like(areas)
    dists = np.zeros_like(extents)

    # get the max distance from the center, area and extent
    center = np.array(depth_image.shape)/2

    for i, props in enumerate(region_properties):
        areas[i] = props.area
        extents[i] = props.extent
        tmp_dists = np.sqrt(np.sum(np.square(props.coords-center), 1))
        dists[i] = tmp_dists.max()

    # rank features
    ranks = np.vstack((scipy.stats.rankdata(-areas, method='max'),
                       scipy.stats.rankdata(-extents, method='max'),
                       scipy.stats.rankdata(dists, method='max')))
    weight_array = np.array(bg_roi_weights, 'float32')
    shape_index = np.mean(np.multiply(ranks.astype('float32'), weight_array[:, np.newaxis]), 0).argsort()

    # expansion microscopy on the roi
    rois = []
    bboxes = []

    # Perform image processing on each found ROI
    for shape in shape_index:
        roi = np.zeros_like(depth_image)
        roi[region_properties[shape].coords[:, 0],
            region_properties[shape].coords[:, 1]] = 1
        if strel_dilate is not None:
            roi = cv2.dilate(roi, strel_dilate, iterations=dilate_iterations) # Dilate
        if strel_erode is not None:
            roi = cv2.erode(roi, strel_erode, iterations=erode_iterations) # Erode
        if bg_roi_fill_holes:
            roi = scipy.ndimage.morphology.binary_fill_holes(roi) # Fill Holes

        rois.append(roi)
        bboxes.append(get_bbox(roi))

    # Remove largest overlapping found ROI
    if overlap_roi is not None:
        overlaps = np.zeros_like(areas)

        for i in range(len(rois)):
            overlaps[i] = np.sum(np.logical_and(overlap_roi, rois[i]))

        del_roi = np.argmax(overlaps)
        del rois[del_roi]
        del bboxes[del_roi]

    if get_all_data == True:
        return rois, roi_plane, bboxes, label_im, ranks, shape_index
    else:
        return rois, roi_plane


def apply_roi(frames, roi):
    """
    Apply ROI to data.

    Args:
    frames (np.ndarray): input frames to apply ROI.
    roi (np.ndarray): selected ROI to extract from input images.

    Returns:
    cropped_frames (np.ndarray): Frames cropped around ROI Bounding Box.
    """

    # yeah so fancy indexing slows us down by 3-5x
    cropped_frames = frames*roi
    bbox = get_bbox(roi)

    cropped_frames = cropped_frames[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]
    return cropped_frames


def im_moment_features(IM):
    """
    Use the method of moments and centralized moments to get image properties.

    Args:
    IM (numpy.ndarray): depth image

    Returns:
    features (dict): returns a dictionary with orientation, centroid, and ellipse axis length
    """

    tmp = cv2.moments(IM)
    num = 2*tmp['mu11']
    den = tmp['mu20']-tmp['mu02']

    common = np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    if tmp['m00'] == 0:
        features = {
            'orientation': np.nan,
            'centroid': np.nan,
            'axis_length': [np.nan, np.nan]}
    else:
        features = {
            'orientation': -.5*np.arctan2(num, den),
            'centroid': [tmp['m10']/tmp['m00'], tmp['m01']/tmp['m00']],
            'axis_length': [2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                            2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
        }

    return features


def clean_frames(frames, prefilter_space=(3,), prefilter_time=None,
                 strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                 iters_tail=None, frame_dtype='uint8',
                 strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                 iters_min=None, progress_bar=False):
    """
    Simple temporal and/or spatial filtering, median filter and morphological opening.

    Args:
    frames (np.ndarray): Frames (frames x rows x columns) to filter.
    prefilter_space (tuple): kernel size for spatial filtering
    prefilter_time (tuple): kernel size for temporal filtering
    strel_tail (cv2.StructuringElement): Element for tail filtering.
    iters_tail (int): number of iterations to run opening
    frame_dtype (str): frame encodings
    strel_min (int): minimum kernel size
    iters_min (int): minimum number of filtering iterations
    progress_bar (bool): display progress bar

    Returns:
    filtered_frames (numpy.ndarray): frames x rows x columns
    """

    # seeing enormous speed gains w/ opencv
    filtered_frames = frames.copy().astype(frame_dtype)

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Cleaning frames'):
        # Erode Frames
        if iters_min is not None and iters_min > 0:
            filtered_frames[i] = cv2.erode(filtered_frames[i], strel_min, iters_min)
        # Median Blur
        if prefilter_space is not None and np.all(np.array(prefilter_space) > 0):
            for j in range(len(prefilter_space)):
                filtered_frames[i] = cv2.medianBlur(filtered_frames[i], prefilter_space[j])
        # Tail Filter
        if iters_tail is not None and iters_tail > 0:
            filtered_frames[i] = cv2.morphologyEx(filtered_frames[i], cv2.MORPH_OPEN, strel_tail, iters_tail)

    # Temporal Median Filter
    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0):
        for j in range(len(prefilter_time)):
            filtered_frames = scipy.signal.medfilt(filtered_frames, [prefilter_time[j], 1, 1])

    return filtered_frames


def get_frame_features(frames, frame_threshold=10, mask=np.array([]),
                       mask_threshold=-30, use_cc=False, progress_bar=False):
    """
    Use image moments to compute features of the largest object in the frame

    Args:
    frames (3d np.ndarray): input frames
    frame_threshold (int): threshold in mm separating floor from mouse
    mask (3d np.ndarray): input frame mask for parts not to filter.
    mask_threshold (int): threshold to include regions into mask.
    use_cc (bool): Use connected components.
    progress_bar (bool): Display progress bar.

    Returns:
    features (dict of lists): dictionary with simple image features
    mask (3d np.ndarray): input frame mask.
    """

    nframes = frames.shape[0]

    # Get frame mask
    if type(mask) is np.ndarray and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), 'uint8')

    # Pack contour features into dict
    features = {
        'centroid': np.full((nframes, 2), np.nan),
        'orientation': np.full((nframes,), np.nan),
        'axis_length': np.full((nframes, 2), np.nan)
    }

    for i in tqdm(range(nframes), disable=not progress_bar, desc='Computing moments'):
        # Threshold frame to compute mask
        frame_mask = frames[i] > frame_threshold

        # Incorporate largest connected component with frame mask
        if use_cc:
            cc_mask = get_largest_cc((frames[[i]] > mask_threshold).astype('uint8')).squeeze()
            frame_mask = np.logical_and(cc_mask, frame_mask)

        # Apply mask
        if has_mask:
            frame_mask = np.logical_and(frame_mask, mask[i] > mask_threshold)
        else:
            mask[i] = frame_mask

        # Get contours in frame
        cnts, hierarchy = cv2.findContours(frame_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        # Get features from contours
        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value

    return features, mask


def crop_and_rotate_frames(frames, features, crop_size=(80, 80), progress_bar=False):
    """
    Crop mouse from image and orients it such that the head is pointing right

    Args:
    frames (3d np.ndarray): frames to crop and rotate
    features (dict): dict of extracted features, found in result_00.h5 files.
    crop_size (tuple): size of cropped image.
    progress_bar (bool): Display progress bar.

    Returns:
    cropped_frames (3d np.ndarray): Crop and rotated frames.
    """

    nframes = frames.shape[0]

    # Prepare cropped frame array
    cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), frames.dtype)

    # Get window dimensions
    win = (crop_size[0] // 2, crop_size[1] // 2 + 1)
    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Rotating'):

        if np.any(np.isnan(features['centroid'][i])):
            continue

        # Get bounded frames
        use_frame = cv2.copyMakeBorder(frames[i], *border, cv2.BORDER_CONSTANT, 0)

        # Get row and column centroids
        rr = np.arange(features['centroid'][i, 1]-win[0],
                       features['centroid'][i, 1]+win[1]).astype('int16')
        cc = np.arange(features['centroid'][i, 0]-win[0],
                       features['centroid'][i, 0]+win[1]).astype('int16')

        rr = rr+crop_size[0]
        cc = cc+crop_size[1]

        # Ensure centroids are in bounded frame
        if (np.any(rr >= use_frame.shape[0]) or np.any(rr < 1)
                or np.any(cc >= use_frame.shape[1]) or np.any(cc < 1)):
            continue

        # Rotate the frame such that the mouse is oriented facing east
        rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2),
                                          -np.rad2deg(features['orientation'][i]), 1)
        cropped_frames[i] = cv2.warpAffine(use_frame[rr[0]:rr[-1], cc[0]:cc[-1]],
                                           rot_mat, (crop_size[0], crop_size[1]))

    return cropped_frames


def _area_px_to_mm2(area_px, px_to_mm):
    """
    Convert a pixel-count area into physical area.

    ``px_to_mm`` holds two independent per-axis scale factors in mm/px, so a
    pixel count converts to an area by the product of both factors, not by a
    single linear factor. Using one linear factor leaves the result in px * mm.

    Args:
    area_px (np.ndarray): nframes, mask area measured in pixels
    px_to_mm (np.ndarray): nframes x 2, per-axis conversion factors in mm/px

    Returns:
    area_mm (np.ndarray): nframes, area in mm^2
    """

    return area_px * px_to_mm[:, 0] * px_to_mm[:, 1]


def compute_scalars(frames, track_features, min_height=10, max_height=100, true_depth=673.1):
    """
    Compute extracted scalars.

    Args:
    frames (np.ndarray): frames x r x c, uncropped mouse
    track_features (dict):  dictionary with tracking variables (centroid and orientation)
    min_height (float): minimum height of the mouse
    max_height (float): maximum height of the mouse
    true_depth (float): detected true depth

    Returns:
    features (dict): dictionary of scalars
    """

    nframes = frames.shape[0]

    # Pack features into dict
    features = {
        'centroid_x_px': np.zeros((nframes,), 'float32'),
        'centroid_y_px': np.zeros((nframes,), 'float32'),
        'velocity_2d_px': np.zeros((nframes,), 'float32'),
        'velocity_3d_px': np.zeros((nframes,), 'float32'),
        'width_px': np.zeros((nframes,), 'float32'),
        'length_px': np.zeros((nframes,), 'float32'),
        'area_px': np.zeros((nframes,)),
        'centroid_x_mm': np.zeros((nframes,), 'float32'),
        'centroid_y_mm': np.zeros((nframes,), 'float32'),
        'velocity_2d_mm': np.zeros((nframes,), 'float32'),
        'velocity_3d_mm': np.zeros((nframes,), 'float32'),
        'width_mm': np.zeros((nframes,), 'float32'),
        'length_mm': np.zeros((nframes,), 'float32'),
        'area_mm': np.zeros((nframes,)),
        'height_ave_mm': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
    }

    # Get mm centroid
    centroid_mm = convert_pxs_to_mm(track_features['centroid'], true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(track_features['centroid'] + 1, true_depth=true_depth)

    # Based on the centroid of the mouse, get the mm_to_px conversion
    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)
    masked_frames = np.logical_and(frames > min_height, frames < max_height)

    features['centroid_x_px'] = track_features['centroid'][:, 0]
    features['centroid_y_px'] = track_features['centroid'][:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    features['width_px'] = np.min(track_features['axis_length'], axis=1)
    features['length_px'] = np.max(track_features['axis_length'], axis=1)
    features['area_px'] = np.sum(masked_frames, axis=(1, 2))

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = _area_px_to_mm2(features['area_px'], px_to_mm)

    features['angle'] = track_features['orientation']

    nmask = np.sum(masked_frames, axis=(1, 2))

    for i in range(nframes):
        if nmask[i] > 0:
            features['height_ave_mm'][i] = np.mean(
                frames[i, masked_frames[i]])

    vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
    vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

    features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_px'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

    features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_mm'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features


def feature_hampel_filter(features, centroid_hampel_span=None, centroid_hampel_sig=3,
                          angle_hampel_span=None, angle_hampel_sig=3):
    """
    Filter computed extraction features using Hampel Filtering.

    Args:
    features (dict): dictionary of video features
    centroid_hampel_span (int): Centroid Hampel Span Filtering Kernel Size
    centroid_hampel_sig (int): Centroid Hampel Signal Filtering Kernel Size
    angle_hampel_span (int): Angle Hampel Span Filtering Kernel Size
    angle_hampel_sig (int): Angle Hampel Span Filtering Kernel Size

    Returns:
    features (dict): filtered version of input dict.
    """
    if centroid_hampel_span is not None and centroid_hampel_span > 0:
        padded_centroids = np.pad(features['centroid'],
                                  (((centroid_hampel_span // 2, centroid_hampel_span // 2)),
                                   (0, 0)),
                                  'constant', constant_values = np.nan)
        for i in range(2):
            vws = strided_app(padded_centroids[:, i], centroid_hampel_span, 1)
            med = np.nanmedian(vws, axis=1)
            mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
            vals = np.abs(features['centroid'][:, i] - med)
            fill_idx = np.where(vals > centroid_hampel_sig * 1.4826 * mad)[0]
            features['centroid'][fill_idx, i] = med[fill_idx]

        padded_orientation = np.pad(features['orientation'],
                                    (angle_hampel_span // 2, angle_hampel_span // 2),
                                    'constant', constant_values = np.nan)

    if angle_hampel_span is not None and angle_hampel_span > 0:
        vws = strided_app(padded_orientation, angle_hampel_span, 1)
        med = np.nanmedian(vws, axis=1)
        mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
        vals = np.abs(features['orientation'] - med)
        fill_idx = np.where(vals > med + angle_hampel_sig * mad)[0]
        features['orientation'][fill_idx] = med[fill_idx]

    return features


def model_smoother(features, ll=None, clips=(-300, -125)):
    """
    Apply spatial feature filtering.

    Args:
    features (dict): dictionary of extraction scalar features
    ll (numpy.array): array of loglikelihoods of pixels in frame
    clips (tuple): tuple to ensure video is indexed properly

    Returns:
    features (dict): smoothed version of input features
    """

    if ll is None or clips is None or (clips[0] >= clips[1]):
        return features

    ave_ll = np.zeros((ll.shape[0], ))
    for i, ll_frame in enumerate(ll):

        max_mu = clips[1]
        min_mu = clips[0]

        smoother = np.mean(ll[i])
        smoother -= min_mu
        smoother /= (max_mu - min_mu)

        smoother = np.clip(smoother, 0, 1)
        ave_ll[i] = smoother

    for k, v in features.items():
        nans = np.isnan(v)
        ndims = len(v.shape)
        xvec = np.arange(len(v))
        if nans.any():
            if ndims == 2:
                for i in range(v.shape[1]):
                    f = scipy.interpolate.interp1d(xvec[~nans[:, i]], v[~nans[:, i], i],
                                                   kind='nearest', fill_value='extrapolate')
                    fill_vals = f(xvec[nans[:, i]])
                    features[k][xvec[nans[:, i]], i] = fill_vals
            else:
                f = scipy.interpolate.interp1d(xvec[~nans], v[~nans],
                                               kind='nearest', fill_value='extrapolate')
                fill_vals = f(xvec[nans])
                features[k][nans] = fill_vals

    for i in range(2, len(ave_ll)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i - 1] + smoother * v[i]

    for i in reversed(range(len(ave_ll) - 1)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i + 1] + smoother * v[i]

    return features

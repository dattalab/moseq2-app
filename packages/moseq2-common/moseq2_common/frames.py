"""Cleaning of depth frames: PCA's and crowd movies' spatial cleaning, and
extraction's cleaning settings.

``CleaningOptions`` declares extraction's cleaning settings, and
``cleaning_arguments`` turns them into the arguments of extraction's
``clean_frames``. Extraction and flip-classifier training both clean through
them, so a classifier is trained on frames cleaned as extraction cleans them.

This module imports OpenCV, which comes with the ``video`` extra:
``python -m pip install "moseq2-common[video]"``. moseq2-extract, moseq2-pca
and moseq2-viz require that extra.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from moseq2_common.options import Options, option

__all__ = ["CleaningOptions", "clean_frames", "cleaning_arguments", "select_strel"]


@dataclass(frozen=True, kw_only=True)
class CleaningOptions(Options):
    """Extraction's frame cleaning: the height threshold, the median filters, and the tail and cable filters."""

    temporal_filter_size: tuple = option((0,), "--temporal-filter-size", "-t", type=int, multiple=True, arena=True,
                                         help="Time prefilter kernel (median filter, must be odd)")
    spatial_filter_size: tuple = option((3,), "--spatial-filter-size", "-s", type=int, multiple=True, arena=True,
                                        help="Space prefilter kernel (median filter, must be odd)")
    tail_filter_shape: str = option("ellipse", "--tail-filter-shape", type=str, arena=True,
                                    help="Tail filter shape")
    tail_filter_size: tuple = option((9, 9), "--tail-filter-size", type=(int, int), arena=True,
                                     help="Tail filter size")
    tail_filter_iters: int = option(1, "--tail-filter-iters", type=int, arena=True,
                                    help="Number of tail filter iterations")
    cable_filter_size: tuple = option((5, 5), "--cable-filter-size", type=(int, int), arena=True,
                                      help="Cable filter size (in pixels)")
    cable_filter_shape: str = option("rectangle", "--cable-filter-shape", type=str, arena=True,
                                     help="Cable filter shape (rectangle or ellipse)")
    cable_filter_iters: int = option(0, "--cable-filter-iters", type=int, arena=True,
                                     help="Number of cable filter iterations")
    min_height: int = option(10, "--min-height", type=int, arena=True,
                             help="Min mouse height threshold from floor (mm)")


def cleaning_arguments(opts: CleaningOptions) -> dict:
    """The arguments of extraction's ``clean_frames`` for the settings ``opts``.

    Args:
    opts (CleaningOptions): the cleaning settings.

    Returns:
    arguments (dict): ``prefilter_space``, ``prefilter_time``, ``strel_tail``,
        ``iters_tail``, ``strel_min`` and ``iters_min``. The height threshold
        (``opts.min_height``) is applied before cleaning and is not among them.
    """
    return {
        "prefilter_space": opts.spatial_filter_size,
        "prefilter_time": opts.temporal_filter_size,
        "strel_tail": select_strel(opts.tail_filter_shape, opts.tail_filter_size),
        "iters_tail": opts.tail_filter_iters,
        "strel_min": select_strel(opts.cable_filter_shape, opts.cable_filter_size),
        "iters_min": opts.cable_filter_iters,
    }


def select_strel(shape: str | None = "e", size: Sequence[int] = (10, 10)) -> np.ndarray | None:
    """Build a structuring element from its shape name and size.

    Args:
    shape (str or None): starts with ``e`` for an ellipse or ``r`` for a
        rectangle; any other name gives an ellipse.
    size (sequence of int): the element's width and height.

    Returns:
    strel (numpy.ndarray or None): the element, or None when ``shape`` is None,
        empty or contains ``none``, or when every entry of ``size`` is zero.
    """
    if not isinstance(size, tuple):
        size = tuple(size)

    if shape is None or 'none' in shape or np.all(np.array(size) == 0) or len(shape) == 0:
        strel = None
    elif shape[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif shape[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return strel


def clean_frames(frames: np.ndarray, *, medfilter_space=None, gaussfilter_space=None,
                 tail_filter=None, tail_threshold=5) -> np.ndarray:
    """Filter spatial noise from each frame. The input is not modified.

    The filters run in this order, each skipped when its argument is None:

    1. the tail filter: pixels where the frame's morphological opening by
       ``tail_filter`` is not above ``tail_threshold`` are set to zero;
    2. a median filter for each kernel size in ``medfilter_space``, in turn
       (skipped unless every size is positive);
    3. a Gaussian filter with a (21, 21) kernel and the x and y sigmas in
       ``gaussfilter_space`` (skipped unless both are positive).

    Args:
    frames (numpy.ndarray): frames to filter, shape (n, height, width).
    medfilter_space (sequence of int): median kernel sizes, each odd.
    gaussfilter_space (sequence of float): x and y sigma of the Gaussian.
    tail_filter (numpy.ndarray): structuring element, as ``select_strel`` builds.
    tail_threshold (int): height a pixel's opening must exceed to be kept.

    Returns:
    out (numpy.ndarray): the filtered frames, same shape and dtype.
    """
    out = np.copy(frames)

    if tail_filter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(out[i], cv2.MORPH_OPEN, tail_filter) > tail_threshold
            out[i] = out[i] * mask.astype(frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                out[i] = cv2.medianBlur(out[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(out[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    return out

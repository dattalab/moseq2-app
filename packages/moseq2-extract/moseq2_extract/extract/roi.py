"""
ROI detection pre-processing utilities for fitting a plane to an input depth image.
"""

import numpy as np
from tqdm.auto import tqdm


def plane_fit3(points):
    """
    Fit a plane to 3 points (min number of points for fitting a plane)

    Args:
    points (numpy.ndarray): each row is a group of points, columns correspond to x,y,z.

    Returns:
    plane (numpy.array): linear plane fit-->a*x+b*y+c*z+d
    """

    a = points[1] - points[0]
    b = points[2] - points[0]
    # cross prod to make sure the three points make an area, hence a plane.
    normal = np.array(
        [
            [a[1] * b[2] - a[2] * b[1]],
            [a[2] * b[0] - a[0] * b[2]],
            [a[0] * b[1] - a[1] * b[0]],
        ]
    ).astype("float")
    denom = np.sum(np.square(normal)).astype("float")
    if denom < np.spacing(1):
        plane = np.empty((4,))
        plane[:] = np.nan
    else:
        normal /= np.sqrt(denom)
        d = np.dot(-points[0], normal)
        plane = np.hstack((normal.flatten(), d))

    return plane


def plane_ransac(
    depth_image,
    bg_roi_depth_range=(650, 750),
    iters=1000,
    noise_tolerance=30,
    in_ratio=0.1,
    progress_bar=False,
    mask=None,
    **kwargs,
):
    """
    Fit a plane using a naive RANSAC implementation

    Args:
    depth_image (numpy.ndarray): background image to fit plane to
    bg_roi_depth_range (tuple): min/max depth (mm) to consider pixels for plane
    iters (int): number of RANSAC iterations
    noise_tolerance (float): distance from plane to consider a point an inlier
    in_ratio (float): fraction of points required to consider a plane fit good
    progress_bar (bool): display progress bar
    mask (numpy.ndarray): boolean mask to find region to use
    kwargs (dict): dictionary containing extra keyword arguments from moseq2_extract.proc.get_roi()

    Returns:
    best_plane (numpy.array): plane fit to data
    dist (numpy.array): distance of the calculated coordinates and "best plane"
    """

    use_points = np.logical_and(
        depth_image > bg_roi_depth_range[0], depth_image < bg_roi_depth_range[1]
    )
    if np.sum(use_points) <= 10:
        raise ValueError(
            f'Too few datapoints exist within given "bg roi depth range" {bg_roi_depth_range} -- data point count: {np.sum(use_points)}.'
            "Please adjust this parameter to fit your recording sessions."
        )

    if mask is not None:
        use_points = np.logical_and(use_points, mask)

    xx, yy = np.meshgrid(
        np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    )

    coords = np.vstack(
        (
            xx[use_points].ravel(),
            yy[use_points].ravel(),
            depth_image[use_points].ravel(),
        )
    )
    coords = coords.T

    best_dist = np.inf
    best_num = 0

    npoints = np.sum(use_points)

    for _ in tqdm(range(iters), disable=not progress_bar, desc="Finding plane"):

        sel = coords[np.random.choice(coords.shape[0], 3, replace=True)]
        tmp_plane = plane_fit3(sel)

        if np.all(np.isnan(tmp_plane)):
            continue

        dist = np.abs(np.dot(coords, tmp_plane[:3]) + tmp_plane[3])
        inliers = dist < noise_tolerance
        ninliers = np.sum(inliers)

        if (
            (ninliers / npoints) > in_ratio
            and ninliers > best_num
            and np.mean(dist) < best_dist
        ):
            best_dist = np.mean(dist)
            best_num = ninliers
            best_plane = tmp_plane

    # fit the plane to our x,y,z coordinates
    coords = np.vstack((xx.ravel(), yy.ravel(), depth_image.ravel())).T
    dist = np.abs(np.dot(coords, best_plane[:3]) + best_plane[3])

    return best_plane, dist

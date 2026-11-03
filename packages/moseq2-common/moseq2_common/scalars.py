"""Pixels to millimetres, and the scalars derived from a mouse's centroid.

Extraction (``moseq2_extract.extract.proc.compute_scalars``) and viz's
conversion of legacy-format scalars (``moseq2_viz.scalars.util.convert_legacy_scalars``)
both convert the centroid to millimetres and derive the velocities here.
"""

import numpy as np

__all__ = [
    "DEPTH_FIELD_OF_VIEW",
    "centroid_to_mm",
    "convert_pxs_to_mm",
    "depth_camera_geometry",
    "velocities",
]

#: Nominal depth field of view in degrees (horizontal, vertical) for each depth frame size a
#: supported camera writes: the Kinect v2, and the Azure Kinect and Orbbec Femto narrow
#: (75 x 65) and wide (120 x 120) modes, binned and unbinned. The wide modes are fisheye, so
#: a pinhole conversion of them is approximate away from the image centre.
DEPTH_FIELD_OF_VIEW = {
    (512, 424): (70.6, 60),
    (640, 576): (75, 65),
    (320, 288): (75, 65),
    (512, 512): (120, 120),
    (1024, 1024): (120, 120),
}


def depth_camera_geometry(resolution):
    """
    Pick the frame size and nominal field of view to convert a recording's pixels to mm with.

    Args:
    resolution (tuple): (width, height) of the raw depth frames

    Returns:
    resolution (tuple): the frame size to convert with; a size no supported camera writes is converted as a Kinect v2 frame
    field_of_view (tuple): nominal (horizontal, vertical) field of view in degrees
    """

    resolution = tuple(int(n) for n in resolution)
    if resolution not in DEPTH_FIELD_OF_VIEW:
        resolution = (512, 424)
    return resolution, DEPTH_FIELD_OF_VIEW[resolution]


def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """
    Convert x, y coordinates in pixel space to mm.

    Args:
    coords (np.ndarray): n x 2 array of x, y pixel coordinates
    resolution (tuple): image dimensions
    field_of_view (tuple): nominal (horizontal, vertical) field of view of the depth camera, in degrees
    true_depth (float): distance between the depth camera and the arena floor, in mm

    Returns:
    new_coords (np.ndarray): n x 2 array of x, y coordinates in mm
    """

    # http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    # http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    # http://smeenk.com/kinect-field-of-view-comparison/

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.tan(np.deg2rad(field_of_view[0] / 2)))
    fh = resolution[1] / (2 * np.tan(np.deg2rad(field_of_view[1] / 2)))

    new_coords = np.zeros_like(coords, dtype=np.float64)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def centroid_to_mm(centroid_px, resolution, field_of_view, true_depth):
    """
    Convert a centroid track to mm, with the mm per pixel at each centroid.

    Args:
    centroid_px (np.ndarray): nframes x 2, x, y centroid in pixels
    resolution (tuple): frame size to convert with
    field_of_view (tuple): nominal (horizontal, vertical) field of view in degrees
    true_depth (float): distance between the depth camera and the arena floor, in mm

    Returns:
    centroid_mm (np.ndarray): nframes x 2, x, y centroid in mm
    px_to_mm (np.ndarray): nframes x 2, mm per pixel along x and y, from the centroid shifted by one pixel
    """

    centroid_mm = convert_pxs_to_mm(centroid_px, resolution=resolution,
                                    field_of_view=field_of_view, true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(centroid_px + 1, resolution=resolution,
                                          field_of_view=field_of_view, true_depth=true_depth)
    return centroid_mm, np.abs(centroid_mm_shift - centroid_mm)


def velocities(centroid_x_px, centroid_y_px, centroid_x_mm, centroid_y_mm, height_ave_mm):
    """
    Frame-to-frame velocities of a centroid track; the first frame's are zero.

    Args:
    centroid_x_px, centroid_y_px (np.ndarray): nframes, centroid in pixels
    centroid_x_mm, centroid_y_mm (np.ndarray): nframes, centroid in mm
    height_ave_mm (np.ndarray): nframes, mean height of the mouse in mm

    Returns:
    velocities (dict): ``velocity_2d_px``, ``velocity_3d_px`` (height in mm), ``velocity_2d_mm``,
        ``velocity_3d_mm`` and ``velocity_theta`` (direction of the mm velocity, in radians)
    """

    vel_x = np.diff(np.concatenate((centroid_x_px[:1], centroid_x_px)))
    vel_y = np.diff(np.concatenate((centroid_y_px[:1], centroid_y_px)))
    vel_z = np.diff(np.concatenate((height_ave_mm[:1], height_ave_mm)))

    result = {
        'velocity_2d_px': np.hypot(vel_x, vel_y),
        'velocity_3d_px': np.sqrt(np.square(vel_x) + np.square(vel_y) + np.square(vel_z)),
    }

    vel_x = np.diff(np.concatenate((centroid_x_mm[:1], centroid_x_mm)))
    vel_y = np.diff(np.concatenate((centroid_y_mm[:1], centroid_y_mm)))

    result['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
    result['velocity_3d_mm'] = np.sqrt(np.square(vel_x) + np.square(vel_y) + np.square(vel_z))
    result['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return result

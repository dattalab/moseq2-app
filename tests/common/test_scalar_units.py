"""Pixels to millimetres and the scalars derived from the centroid, in moseq2-common.

Extraction's ``compute_scalars`` and viz's ``convert_legacy_scalars`` each had
their own ``convert_pxs_to_mm`` and their own block turning a centroid into
millimetres and velocities. Both use ``moseq2_common.scalars`` now: the depth
camera field-of-view table, the pinhole conversion, the centroid's mm and
per-axis mm per pixel, and the five velocities.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_common.scalars import (
    DEPTH_FIELD_OF_VIEW,
    centroid_to_mm,
    convert_pxs_to_mm,
    depth_camera_geometry,
    velocities,
)

DEPTH = 500.0


@pytest.mark.parametrize("resolution", [(512, 424), (640, 576), (512, 512)])
def test_the_frame_centre_converts_to_zero(resolution):
    centre = np.array([[resolution[0] // 2, resolution[1] // 2]], dtype=float)
    _, field_of_view = depth_camera_geometry(resolution)
    np.testing.assert_array_equal(
        convert_pxs_to_mm(centre, resolution=resolution, field_of_view=field_of_view, true_depth=DEPTH),
        [[0.0, 0.0]],
    )


@pytest.mark.parametrize(
    "resolution, field_of_view",
    [
        ((512, 424), (70.6, 60)),
        ((640, 576), (75, 65)),
        ((320, 288), (75, 65)),
        ((512, 512), (120, 120)),
        ((1024, 1024), (120, 120)),
    ],
    ids=["Kinect v2", "Azure narrow", "Azure narrow binned", "Azure wide binned", "Azure wide"],
)
def test_each_camera_frame_size_has_its_field_of_view(resolution, field_of_view):
    assert DEPTH_FIELD_OF_VIEW[resolution] == field_of_view
    assert depth_camera_geometry(resolution) == (resolution, field_of_view)
    # the size may arrive as a numpy array or list
    assert depth_camera_geometry(np.array(resolution)) == (resolution, field_of_view)


def test_an_unknown_frame_size_converts_as_a_kinect_v2_frame():
    assert depth_camera_geometry((640, 480)) == ((512, 424), (70.6, 60))


def test_centroid_to_mm_gives_the_scale_one_pixel_away():
    centroid = np.array([[100.0, 50.0], [300.5, 400.25]])
    resolution, field_of_view = (640, 576), (75, 65)

    centroid_mm, px_to_mm = centroid_to_mm(centroid, resolution, field_of_view, DEPTH)

    np.testing.assert_array_equal(
        centroid_mm, convert_pxs_to_mm(centroid, resolution, field_of_view, DEPTH)
    )
    np.testing.assert_array_equal(
        px_to_mm,
        np.abs(convert_pxs_to_mm(centroid + 1, resolution, field_of_view, DEPTH) - centroid_mm),
    )
    # a pinhole camera has the same scale everywhere on a plane at one depth
    expected = DEPTH * 2 * np.tan(np.deg2rad(np.array(field_of_view) / 2)) / np.array(resolution)
    np.testing.assert_allclose(px_to_mm, np.tile(expected, (2, 1)))


def test_velocities_start_at_zero_and_follow_the_centroid():
    x_px = np.array([10.0, 13.0, 13.0])
    y_px = np.array([20.0, 24.0, 24.0])
    x_mm = 2 * x_px
    y_mm = 2 * y_px
    height = np.array([30.0, 30.0, 32.0])

    v = velocities(x_px, y_px, x_mm, y_mm, height)

    assert set(v) == {"velocity_2d_px", "velocity_3d_px", "velocity_2d_mm", "velocity_3d_mm", "velocity_theta"}
    np.testing.assert_array_equal(v["velocity_2d_px"], [0.0, 5.0, 0.0])
    np.testing.assert_array_equal(v["velocity_3d_px"], [0.0, 5.0, 2.0])
    np.testing.assert_array_equal(v["velocity_2d_mm"], [0.0, 10.0, 0.0])
    np.testing.assert_array_equal(v["velocity_3d_mm"], [0.0, 10.0, 2.0])
    np.testing.assert_allclose(v["velocity_theta"], [0.0, np.arctan2(8.0, 6.0), 0.0])

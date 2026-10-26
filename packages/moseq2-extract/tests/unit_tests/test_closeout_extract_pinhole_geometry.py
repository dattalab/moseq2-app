"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


def test_nominal_fov_maps_image_edge_to_pinhole_extent():
    from moseq2_extract.util import convert_pxs_to_mm
    coords = np.array([[512., 424.], [256., 212.]])
    actual = convert_pxs_to_mm(coords, true_depth=500.)
    np.testing.assert_allclose(actual[0], 500 * np.tan(np.deg2rad([35.3, 30.])))
    np.testing.assert_array_equal(actual[1], [0., 0.])

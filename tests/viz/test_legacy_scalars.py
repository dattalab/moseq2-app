"""Converting scalars written by older extractions.

``scalars_to_dataframe`` hands an extraction's scalars to
``convert_legacy_scalars`` when they need converting, or always when asked to
force it. Older extractions name the pixel measurements ``width``, ``length``
and ``area``; current ones name them ``width_px``, ``length_px`` and
``area_px``.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_viz.scalars.util import convert_legacy_scalars, convert_pxs_to_mm


@pytest.mark.parametrize("suffix", ["", "_px"], ids=["older names", "current names"])
def test_pixel_measurements_are_read_under_either_name(suffix):
    """Whichever name the extraction wrote, the measurement is read from it."""
    scalars = {key: np.ones(3) for key in ["centroid_x", "centroid_y", "height_ave", "angle"]}
    for key in ["width", "length", "area"]:
        scalars[key + suffix] = np.array([3.0, 4.0, 5.0])

    converted = convert_legacy_scalars(scalars, force=True)

    for key in ["width", "length", "area"]:
        np.testing.assert_array_equal(converted[key + "_px"], [3.0, 4.0, 5.0])


def test_integer_pixels_give_the_same_millimetres_as_float_ones():
    """The conversion's own copy of ``convert_pxs_to_mm`` keeps fractions too."""
    pixels = np.array([[257, 213], [258, 214]], dtype=np.int64)
    millimetres = convert_pxs_to_mm(pixels)

    assert np.issubdtype(millimetres.dtype, np.floating)
    np.testing.assert_allclose(millimetres, convert_pxs_to_mm(pixels.astype(float)))


def test_area_grows_with_the_square_of_depth():
    """An area in mm² converts with the product of the two per-axis factors.

    Moving the same pixel footprint twice as far from the camera doubles each
    side in millimetres, so the area quadruples while the length doubles. This
    matches what extraction computes for ``area_mm``.
    """
    scalars = {
        key: np.array([value], dtype=float)
        for key, value in dict(
            centroid_x=250, centroid_y=210, width=10, length=20, area=150, height_ave=10, angle=0
        ).items()
    }
    near = convert_legacy_scalars(scalars, force=True, true_depth=400)
    far = convert_legacy_scalars(scalars, force=True, true_depth=800)

    np.testing.assert_allclose(far["area_mm"], 4 * near["area_mm"])
    np.testing.assert_allclose(far["length_mm"], 2 * near["length_mm"])

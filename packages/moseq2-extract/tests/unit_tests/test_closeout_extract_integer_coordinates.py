"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


def test_integer_coordinates_preserve_fractional_millimetres():
    from moseq2_extract.util import convert_pxs_to_mm
    coords = np.array([[257, 213], [258, 214]], dtype=np.int64)
    actual = convert_pxs_to_mm(coords)
    np.testing.assert_allclose(actual, convert_pxs_to_mm(coords.astype(float)))
    assert np.issubdtype(actual.dtype, np.floating)

"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd





@pytest.mark.parametrize('pixel_keys', [False, True])
def test_legacy_conversion_accepts_multirow_scalar_mappings(pixel_keys):
    from moseq2_viz.scalars.util import convert_legacy_scalars
    values = {key: np.ones(3) for key in ['centroid_x', 'centroid_y', 'height_ave', 'angle']}
    for key in ['width', 'length', 'area']:
        values[key + ('_px' if pixel_keys else '')] = np.array([3., 4., 5.])
    result = convert_legacy_scalars(values, force=True)
    for key in ['width', 'length', 'area']:
        np.testing.assert_array_equal(result[key + '_px'], [3., 4., 5.])

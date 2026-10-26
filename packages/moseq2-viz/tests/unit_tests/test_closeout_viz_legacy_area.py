"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd





def test_legacy_area_scales_quadratically_with_depth():
    from moseq2_viz.scalars.util import convert_legacy_scalars
    features = {key: np.array([value], dtype=float) for key, value in
                dict(centroid_x=250, centroid_y=210, width=10, length=20, area=150, height_ave=10, angle=0).items()}
    a = convert_legacy_scalars(features, force=True, true_depth=400)
    b = convert_legacy_scalars(features, force=True, true_depth=800)
    np.testing.assert_allclose(b['area_mm'], 4 * a['area_mm'])
    np.testing.assert_allclose(b['length_mm'], 2 * a['length_mm'])

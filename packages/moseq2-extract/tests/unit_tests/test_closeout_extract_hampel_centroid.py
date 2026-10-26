"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


@pytest.mark.parametrize("axis", [0, 1])
def test_hampel_centroid_filter_is_translation_invariant(axis):
    from moseq2_extract.extract.proc import feature_hampel_filter
    x = np.random.RandomState(7).normal(0, .3, (101, 2))
    x[50, axis] = 12
    near = feature_hampel_filter({"centroid": x.copy(), "orientation": np.zeros(101)},
                                centroid_hampel_span=5, centroid_hampel_sig=3, angle_hampel_span=0)["centroid"]
    far = feature_hampel_filter({"centroid": x.copy() + 200, "orientation": np.zeros(101)},
                               centroid_hampel_span=5, centroid_hampel_sig=3, angle_hampel_span=0)["centroid"]
    np.testing.assert_allclose(near, far - 200, atol=1e-12)
    assert abs(near[50, axis]) < 2

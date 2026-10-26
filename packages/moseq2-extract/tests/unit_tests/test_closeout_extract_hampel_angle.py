"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


def test_angle_hampel_filter_can_run_without_centroid_filter():
    from moseq2_extract.extract.proc import feature_hampel_filter
    angles = np.random.RandomState(3).normal(.1, .01, 101)
    angles[50] = 2
    actual = feature_hampel_filter({"centroid": np.zeros((101, 2)), "orientation": angles},
                                  centroid_hampel_span=0, angle_hampel_span=5,
                                  angle_hampel_sig=3)
    assert actual["orientation"][50] < 1

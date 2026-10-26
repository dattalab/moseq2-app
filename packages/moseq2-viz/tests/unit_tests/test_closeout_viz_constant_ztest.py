"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_ztest_equal_constants_are_not_nan():
    from moseq2_viz.model.stat import ztest_vect
    np.testing.assert_array_equal(ztest_vect(np.zeros((20, 2)), np.zeros((20, 2))), [1., 1.])

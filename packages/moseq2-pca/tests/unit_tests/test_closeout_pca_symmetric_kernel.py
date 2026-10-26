"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_gaussian_kernel_is_symmetric():
    from moseq2_pca.util import gaussian_kernel1d
    kernel = gaussian_kernel1d(sig=2.)
    assert len(kernel) % 2 == 1
    np.testing.assert_allclose(kernel, kernel[::-1])


def test_gaussian_impulse_keeps_its_time_origin():
    from moseq2_pca.util import gauss_smooth
    x = np.zeros(101)
    x[50] = 1
    actual = gauss_smooth(x, sig=2)
    assert np.argmax(actual) == 50
    assert np.dot(np.arange(101), actual) == pytest.approx(50.)

"""The changepoint curve with temporal smoothing turned off.

Setting ``sigma`` to 0 (or leaving it unset) in the config skips the Gaussian
smoothing of the projections. Every frame that has a squared difference across
the ``k``-frame lag gets a value; the first ``k // 2`` and last ``k - k // 2``
frames have none, so they stay blank rather than holding a raw score that
would set the baseline the peak threshold is measured from.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_pca.util import get_changepoints

SCORES = np.sin(np.arange(120)[None, :] / 6)
K = 5


@pytest.mark.parametrize("sigma", [0, None])
def test_turning_smoothing_off_keeps_the_curve(sigma):
    """Every frame with a squared difference has a value."""
    _, curve = get_changepoints(SCORES.copy(), k=K, sigma=sigma)
    assert np.isfinite(curve[K // 2:-(K - K // 2)]).all()


@pytest.mark.parametrize("sigma", [0, None])
def test_frames_without_a_squared_difference_are_blank(sigma):
    """The first k // 2 and last k - k // 2 frames hold no value."""
    _, curve = get_changepoints(SCORES.copy(), k=K, sigma=sigma)
    assert np.isnan(curve[:K // 2]).all()
    assert np.isnan(curve[-(K - K // 2):]).all()
